import argparse
import os
import torch

from utils import NoiseScheduleVP, stable_diffusion_beta_schedule, amortize
import libs.autoencoder as autoencoder
from custom_uvit import CustomUviT
from libs.uvit import myUViT
from la_train_datasets import imagenet_feature_dataset
import torchvision.utils as tvu
import tqdm
from pytorch_lightning import seed_everything
from utils import get_model_input_time, sample_from_gaussion, inverse_data_transform, \
singlestep_dpm_solver_second_update, exp_iteration, var_iteration

def conditioned_exp_iteration(exp_xt, ns, s, t, pre_wuq, exp_s1=None, mc_eps_exp_s1= None):

    if pre_wuq == True:
        exp_xt_next = exp_iteration(exp_xt, ns, s, t, mc_eps_exp_s1)
        return exp_xt_next
    else:
        exp_xt_next = exp_iteration(exp_xt, ns, s, t, exp_s1)
        return exp_xt_next

def conditioned_var_iteration(var_xt, ns, s, t, pre_wuq, cov_xt_epst= None, var_epst = None):

    if pre_wuq == True:
        var_xt_next = var_iteration(var_xt, ns, s, t, cov_xt_epst, var_epst)
        return var_xt_next
    else:
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        var_xt_next = torch.square(torch.exp(log_alpha_t - log_alpha_s)) * var_xt
        return var_xt_next

def conditioned_update(ns, x, s, t, custom_uvit, model_s, pre_wuq, r1=0.5, **model_kwargs):
    
    if pre_wuq == True:
        return singlestep_dpm_solver_second_update(ns, x, s, t, custom_uvit, model_s, r1=0.5, **model_kwargs)
    else:
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t)
        sigma_s1, sigma_t = ns.marginal_std(s1), ns.marginal_std(t)

        phi_11 = torch.expm1(r1 * h)
        phi_1 = torch.expm1(h)
        
        x_s1 = (
            torch.exp(log_alpha_s1 - log_alpha_s) * x
            - (sigma_s1 * phi_11) * model_s
        )

        input_s1 = get_model_input_time(ns, s1)
        model_s1 = custom_uvit.accurate_forward(x_s1, input_s1.expand(x_s1.shape[0]), **model_kwargs)

        x_t = (
            torch.exp(log_alpha_t - log_alpha_s) * x
            - (sigma_t * phi_1) * model_s
            - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s)
        )

        return x_t, model_s1
    
def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--seed", type=int, default=1233, help="Random seed")
    parser.add_argument(
        "--skip_type",
        type=str,
        default="time_uniform",
        help="skip according to ('uniform' or 'quadratic' for DDIM/DDPM; 'logSNR' or 'time_uniform' or 'time_quadratic' for DPM-Solver)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--fixed_class", type=int, default=None, help="fixed class label for conditional sampling"
    )
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--denoise", action="store_true", default=False)
    parser.add_argument("--lower_order_final", action="store_true", default=False)
    parser.add_argument("--thresholding", action="store_true", default=False)
    
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--train_la_batch_size", type=int, default=32)

    parser.add_argument("--mc_size", type=int, default=10)
    parser.add_argument("--sample_batch_size", type=int, default=16)
    parser.add_argument("--train_la_data_size", type=int, default=50)
    parser.add_argument("--total_n_sample", type=int, default=50)
    parser.add_argument("--encoder_path", type=str)
    parser.add_argument("--uvit_path", type=str)
    args = parser.parse_args()
    
    if args.config == "imagenet256_uvit_huge.py":
        from configs import imagenet256_uvit_huge
        config = imagenet256_uvit_huge.get_config()
    else:
        from configs import imagenet512_uvit_huge
        config = imagenet512_uvit_huge.get_config()        

    return args, config

def main():
    args, config = parse_args_and_config()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True
    print(args.seed)
    seed_everything(args.seed)

    image_size = config.dataset.image_size
    z_size = image_size // 8
    patch_size = 2 if image_size == 256 else 4
    total_n_samples = args.total_n_sample
    if total_n_samples % args.sample_batch_size != 0:
        raise ValueError("Total samples for sampling must be divided exactly by args.sample_batch_size, but got {} and {}".format(total_n_samples, args.sample_batch_size))
    n_rounds = total_n_samples // args.sample_batch_size
    fixed_xT = torch.randn([args.total_n_sample, 4, z_size, z_size])
    if args.fixed_class == 10000:
        fixed_classes = torch.randint(low=0, high=1000, size=(args.sample_batch_size, n_rounds))
    else:
        fixed_classes = torch.randint(low=args.fixed_class, high=args.fixed_class+1, size=(args.sample_batch_size, n_rounds)).to(device)
    
    ae = autoencoder.get_model(args.encoder_path)
    ae.to(device)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return ae.decode(_batch)
    
    nnet = myUViT(img_size=z_size,
        patch_size=patch_size,
        in_chans=4,
        embed_dim=1152,
        depth=28,
        num_heads=16,
        num_classes=1001,
        conv=False)

    nnet.to(device)
    nnet.load_state_dict(torch.load(args.uvit_path, map_location={'cuda:%d' % 0: 'cuda:%d' % args.device}))
    nnet.eval()
    train_dataset= imagenet_feature_dataset(args = args, config = config, ae = ae)
    train_dataloader= torch.utils.data.DataLoader(train_dataset, batch_size=args.train_la_batch_size, shuffle=True)
    custom_uvit = CustomUviT(nnet, train_dataloader, args, config)

    #########   get t sequence (note that t is different from timestep)  ##########
    betas = stable_diffusion_beta_schedule()
    ns = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(betas, device=device).float())
    def get_time_steps(skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps, **recommended for DPM-Solver**.
                - 'time_uniform': uniform time for the time steps. (Used in DDIM and DDPM.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            lambda_T = ns.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = ns.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            # print(torch.min(torch.abs(logSNR_steps - self.noise_schedule.marginal_lambda(self.noise_schedule.inverse_lambda(logSNR_steps)))).item())
            return ns.inverse_lambda(logSNR_steps)
        elif skip_type == 't2':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t = torch.linspace(t_0, t_T, 10000000).to(device)
            quadratic_t = torch.sqrt(t)
            quadratic_steps = torch.linspace(quadratic_t[0], quadratic_t[-1], N + 1).to(device)
            return torch.flip(torch.cat([t[torch.searchsorted(quadratic_t, quadratic_steps)[:-1]], t_T * torch.ones((1,)).to(device)], dim=0), dims=[0])
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))
    t_0 = 1. / ns.total_N
    t_T = ns.T
    t_seq = get_time_steps(skip_type=args.skip_type, t_T=t_T, t_0=t_0, N=args.timesteps//2, device=device)
    t_seq = torch.flip(t_seq, dims=[0])

    #########   get skip UQ rules  ##########  
    # if uq_array[i] == False, then we use origin_dpmsolver_update from t_seq[i] to t_seq[i-1]
    uq_array = [False] * (args.timesteps//2+1)
    for i in range(args.timesteps//2, 0, -5):
        uq_array[i] = True
    
    #########   start sample  ##########
    exp_dir = f'../exp/imagenet{image_size}/dpmUQ_fixed_class{args.fixed_class}_train%{args.train_la_data_size}_step{args.timesteps}_S{args.mc_size}/'
    os.makedirs(exp_dir, exist_ok=True)
    var_sum = torch.zeros((args.sample_batch_size, n_rounds)).to(device)
    img_id = 1000000
    sample_x = []
    samle_batch_size = args.sample_batch_size
    with torch.no_grad():
        for loop in tqdm.tqdm(
            range(n_rounds), desc="Generating image samples for FID evaluation."
        ):  

            xT = fixed_xT[loop*args.sample_batch_size:(loop+1)*args.sample_batch_size, :, :, :].to(device)
            classes = fixed_classes[:, loop].to(device)
            model_kwargs = {"y": classes}
            timestep, mc_sample_size  = args.timesteps//2, args.mc_size

            ###### Initialize
            T = t_seq[timestep]
            if uq_array[timestep] == True:
                xt_next = xT
                exp_xt_next, var_xt_next = xT, torch.zeros_like(xT).to(device)
                eps_mu_t_next, eps_var_t_next = custom_uvit(xT, get_model_input_time(ns, T).expand(xT.shape[0]), **model_kwargs) 
                cov_xt_next_epst_next = torch.zeros_like(xT).to(device)
                _, model_s1, _ = conditioned_update(ns, xt_next, T, t_seq[timestep-1], custom_uvit, eps_mu_t_next, pre_wuq=True, r1=0.5, **model_kwargs)
                list_eps_mu_t_next_i = torch.unsqueeze(model_s1, dim=0)
            else:
                xt_next = xT
                exp_xt_next, var_xt_next = xT, torch.zeros_like(xT).to(device)
                eps_mu_t_next = custom_uvit.accurate_forward(xT, get_model_input_time(ns, T).expand(xT.shape[0]), **model_kwargs)
            
            ####### Start skip UQ sampling  ######
            for timestep in range(args.timesteps//2, 0, -1):

                if uq_array[timestep] == True:
                    xt = xt_next
                    exp_xt, var_xt = exp_xt_next, var_xt_next
                    eps_mu_t, eps_var_t, cov_xt_epst = eps_mu_t_next, eps_var_t_next, cov_xt_next_epst_next
                    mc_eps_exp_t = torch.mean(list_eps_mu_t_next_i, dim=0)
                else: 
                    xt = xt_next
                    exp_xt, var_xt = exp_xt_next, var_xt_next
                    eps_mu_t = eps_mu_t_next
                
                s, t = t_seq[timestep], t_seq[timestep-1]
                if uq_array[timestep] == True:
                    eps_t= sample_from_gaussion(eps_mu_t, eps_var_t)
                    xt_next, _ , model_s1_var = conditioned_update(ns=ns, x=xt, s=s, t=t, custom_uvit=custom_uvit, model_s=eps_t, pre_wuq=uq_array[timestep], r1=0.5, **model_kwargs)
                    exp_xt_next = conditioned_exp_iteration(exp_xt, ns, s, t, pre_wuq=uq_array[timestep], mc_eps_exp_s1=mc_eps_exp_t)
                    var_xt_next = conditioned_var_iteration(var_xt, ns, s, t, pre_wuq=uq_array[timestep], cov_xt_epst= cov_xt_epst, var_epst=model_s1_var)
                    # decide whether to see xt_next as a random variable
                    if uq_array[timestep-1] == True:
                        list_xt_next_i, list_eps_mu_t_next_i=[], []
                        s_next = t_seq[timestep-1]
                        t_next = t_seq[timestep-2]
                        lambda_s_next, lambda_t_next = ns.marginal_lambda(s_next), ns.marginal_lambda(t_next)
                        h_next = lambda_t_next - lambda_s_next
                        lambda_s1_next = lambda_s_next + 0.5 * h_next
                        s1_next = ns.inverse_lambda(lambda_s1_next)
                        sigma_s1_next = ns.marginal_std(s1_next)
                        log_alpha_s_next, log_alpha_s1_next = ns.marginal_log_mean_coeff(s_next), ns.marginal_log_mean_coeff(s1_next)
                        phi_11_next = torch.expm1(0.5*h_next)

                        for _ in range(mc_sample_size):
                            
                            var_xt_next = torch.clamp(var_xt_next, min=0)
                            xt_next_i = sample_from_gaussion(exp_xt_next, var_xt_next)
                            list_xt_next_i.append(xt_next_i)
                            model_t_i, model_t_i_var = custom_uvit(xt_next_i, get_model_input_time(ns, s_next).expand(xt_next_i.shape[0]), **model_kwargs)
                            xu_next_i = sample_from_gaussion(torch.exp(log_alpha_s1_next - log_alpha_s_next) * xt_next_i-(sigma_s1_next * phi_11_next) * model_t_i, \
                                                            torch.square(sigma_s1_next * phi_11_next) * model_t_i_var)
                            model_u_i, _ = custom_uvit(xu_next_i, get_model_input_time(ns, s1_next).expand(xt_next_i.shape[0]), **model_kwargs)
                            list_eps_mu_t_next_i.append(model_u_i)

                        eps_mu_t_next, eps_var_t_next = custom_uvit(xt_next, get_model_input_time(ns, s_next).expand(xt_next.shape[0]), **model_kwargs)
                        list_xt_next_i = torch.stack(list_xt_next_i, dim=0).to(device)
                        list_eps_mu_t_next_i = torch.stack(list_eps_mu_t_next_i, dim=0).to(device)
                        cov_xt_next_epst_next = torch.mean(list_xt_next_i*list_eps_mu_t_next_i, dim=0)-exp_xt_next*torch.mean(list_eps_mu_t_next_i, dim=0)
                    else:
                        eps_mu_t_next = custom_uvit.accurate_forward(xt_next, get_model_input_time(ns, t).expand(xt_next.shape[0]), **model_kwargs)

                else:
                    xt_next, model_s1 = conditioned_update(ns=ns, x=xt, s=s, t=t, custom_uvit=custom_uvit, model_s=eps_mu_t, pre_wuq=uq_array[timestep], r1=0.5, **model_kwargs)
                    exp_xt_next = conditioned_exp_iteration(exp_xt, ns, s, t, exp_s1= model_s1, pre_wuq=uq_array[timestep])
                    var_xt_next = conditioned_var_iteration(var_xt, ns, s, t, pre_wuq=uq_array[timestep])
                    if uq_array[timestep-1] == True:
                        list_xt_next_i, list_eps_mu_t_next_i=[], []
                        s_next = t_seq[timestep-1]
                        t_next = t_seq[timestep-2]
                        lambda_s_next, lambda_t_next = ns.marginal_lambda(s_next), ns.marginal_lambda(t_next)
                        h_next = lambda_t_next - lambda_s_next
                        lambda_s1_next = lambda_s_next + 0.5 * h_next
                        s1_next = ns.inverse_lambda(lambda_s1_next)
                        sigma_s1_next = ns.marginal_std(s1_next)
                        log_alpha_s_next, log_alpha_s1_next = ns.marginal_log_mean_coeff(s_next), ns.marginal_log_mean_coeff(s1_next)
                        phi_11_next = torch.expm1(0.5*h_next)

                        for _ in range(mc_sample_size):
                            
                            var_xt_next = torch.clamp(var_xt_next, min=0)
                            xt_next_i = sample_from_gaussion(exp_xt_next, var_xt_next)
                            list_xt_next_i.append(xt_next_i)
                            model_t_i, model_t_i_var = custom_uvit(xt_next_i, get_model_input_time(ns, s_next).expand(xt_next_i.shape[0]), **model_kwargs)
                            xu_next_i = sample_from_gaussion(torch.exp(log_alpha_s1_next - log_alpha_s_next) * xt_next_i-(sigma_s1_next * phi_11_next) * model_t_i, \
                                                            torch.square(sigma_s1_next * phi_11_next) * model_t_i_var)
                            model_u_i, _ = custom_uvit(xu_next_i, get_model_input_time(ns, s1_next).expand(xt_next_i.shape[0]), **model_kwargs)
                            list_eps_mu_t_next_i.append(model_u_i)

                        eps_mu_t_next, eps_var_t_next = custom_uvit(xt_next, get_model_input_time(ns, s_next).expand(xt_next.shape[0]), **model_kwargs)
                        list_xt_next_i = torch.stack(list_xt_next_i, dim=0).to(device)
                        list_eps_mu_t_next_i = torch.stack(list_eps_mu_t_next_i, dim=0).to(device)
                        cov_xt_next_epst_next = torch.mean(list_xt_next_i*list_eps_mu_t_next_i, dim=0)-exp_xt_next*torch.mean(list_eps_mu_t_next_i, dim=0)
                    else:
                        eps_mu_t_next = custom_uvit.accurate_forward(xt_next, get_model_input_time(ns, t).expand(xt_next.shape[0]), **model_kwargs)

            ###### Save variance and sample image  ######         
            var_sum[:, loop] = var_xt_next.sum(dim=(1,2,3))
            def decode_large_batch(_batch):
                if z_size == 32:
                    decode_mini_batch_size = 8  # use a small batch size since the decoder is large
                else:
                    decode_mini_batch_size = 1  # use a small batch size since the decoder is large
                xs = []
                pt = 0
                for _decode_mini_batch_size in amortize(_batch.size(0), decode_mini_batch_size):
                    x = decode(_batch[pt: pt + _decode_mini_batch_size])
                    pt += _decode_mini_batch_size
                    xs.append(x)
                xs = torch.concat(xs, dim=0)
                assert xs.size(0) == _batch.size(0)
                return xs
            x = inverse_data_transform(decode_large_batch(xt_next))
            sample_x.append(x)          

        sample_x = torch.concat(sample_x, dim=0)
        var = []
        for j in range(n_rounds):
            var.append(var_sum[:, j])
        var = torch.concat(var, dim=0)
        sorted_var, sorted_indices = torch.sort(var, descending=True)
        reordered_sample_x = torch.index_select(sample_x, dim=0, index=sorted_indices.int())
        grid_sample_x = tvu.make_grid(reordered_sample_x, nrow=8, padding=2)
        tvu.save_image(grid_sample_x.cpu().float(), os.path.join(exp_dir, "sorted_sample.png"))
        

if __name__ == "__main__":
    main()
