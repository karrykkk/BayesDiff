import argparse, os
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

import torchvision.utils as tvu
from ldm.util import instantiate_from_config
from utils import NoiseScheduleVP, get_model_input_time, sample_from_gaussion, \
singlestep_dpm_solver_second_update, exp_iteration, var_iteration
from custom_ld import CustomLD
from dataset import laion_dataset

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

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

def conditioned_update(ns, x, s, t, custom_ld, model_s, pre_wuq, c, r1=0.5):
    
    if pre_wuq == True:
        return singlestep_dpm_solver_second_update(ns, x, s, t, custom_ld, model_s, c, r1=0.5)
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
        model_s1 = custom_ld.accurate_forward(x_s1, input_s1.expand(x_s1.shape[0]), c)

        x_t = (
            torch.exp(log_alpha_t - log_alpha_s) * x
            - (sigma_t * phi_1) * model_s
            - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s)
        )

        return x_t, model_s1
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="Children playing football",
        help="the prompt to render"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="time_uniform",
        help="skip according to ('uniform' or 'quadratic' for DDIM/DDPM; 'logSNR' or 'time_uniform' or 'time_quadratic' for DPM-Solver)",
    )
    parser.add_argument(
        "--laion_art_path",
        type=str,)
    parser.add_argument(
        "--local_image_path",
        type=str,)
    parser.add_argument("--mc_size", type=int, default=3)
    parser.add_argument("--sample_batch_size", type=int, default=8)
    parser.add_argument("--train_la_batch_size", type=int, default=4)
    parser.add_argument("--train_la_data_size", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default= 50)
    parser.add_argument('--device', type=int, default=0)
    opt = parser.parse_args()
    print(opt)
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    # print(model.model.diffusion_model.out[2])
    # Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    train_dataset= laion_dataset(model, opt)
    train_dataloader= torch.utils.data.DataLoader(train_dataset, batch_size=opt.train_la_batch_size, shuffle=False)
    custom_ld = CustomLD(model, train_dataloader)

    #########   get t sequence (note that t is different from timestep)  ##########
    ns = NoiseScheduleVP('discrete', alphas_cumprod=model.alphas_cumprod)
    def get_time_steps(skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
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
            return ns.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_orders_and_timesteps_for_singlestep_solver(steps, t_T, t_0, skip_type, order=2, device=opt.device):

        if steps % 2 == 0:
            K = steps // 2
            orders = [2,] * K
        else:
            K = steps // 2 + 1
            orders = [2,] * (K - 1) + [1]
        
        if skip_type == 'logSNR':
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = get_time_steps(skip_type, t_T, t_0, K, device=opt.device)
        else:
            timesteps_outer = get_time_steps(skip_type, t_T, t_0, steps, device=opt.device)[torch.cumsum(torch.tensor([0,] + orders), 0).to(device)]

        return timesteps_outer
    
    t_0 = 1. / ns.total_N
    t_T = ns.T
    t_seq = get_orders_and_timesteps_for_singlestep_solver(steps= opt.timesteps, t_T=t_T, t_0=t_0, skip_type=opt.skip_type)
    t_seq = torch.flip(t_seq, dims=[0])

    #########   get skip UQ rules  ##########  
    # if uq_array[i] == False, then we use origin_dpmsolver_update from t_seq[i] to t_seq[i-1]
    uq_array = [False] * (opt.timesteps//2+1)
    for i in range(opt.timesteps//2, 0, -2):
        uq_array[i] = True
    print(uq_array)

    #########   start sample  ########## 
    c = model.get_learned_conditioning(opt.prompt)
    c = torch.concat(opt.sample_batch_size * [c], dim=0)
    exp_dir = f'./dpm_solver_2_exp/skipUQ/{opt.prompt}_train{opt.train_la_data_size}_step{opt.timesteps}_S{opt.mc_size}/'
    os.makedirs(exp_dir, exist_ok=True)
    total_n_samples = 96
    if total_n_samples % opt.sample_batch_size != 0:
        raise ValueError("Total samples for sampling must be divided exactly by opt.sample_batch_size, but got {} and {}".format(total_n_samples, opt.sample_batch_size))
    n_rounds = total_n_samples // opt.sample_batch_size
    var_sum = torch.zeros((opt.sample_batch_size, n_rounds)).to(device)
    sample_x = []
    img_id = 1000000
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for loop in tqdm(
                    range(n_rounds), desc="Generating image samples for FID evaluation."
                ):
                    
                    xT, timestep, mc_sample_size  = torch.randn([opt.sample_batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device), opt.timesteps//2, opt.mc_size
                    T = t_seq[timestep]
                    if uq_array[timestep] == True:
                        xt_next = xT
                        exp_xt_next, var_xt_next = xT, torch.zeros_like(xT).to(device)
                        eps_mu_t_next, eps_var_t_next = custom_ld(xT, get_model_input_time(ns, T).expand(xT.shape[0]), c=c) 
                        cov_xt_next_epst_next = torch.zeros_like(xT).to(device)
                        _, model_s1, _ = conditioned_update(ns, xt_next, T, t_seq[timestep-1], custom_ld, eps_mu_t_next, pre_wuq=True, r1=0.5, c=c)
                        list_eps_mu_t_next_i = torch.unsqueeze(model_s1, dim=0)
                    else:
                        xt_next = xT
                        exp_xt_next, var_xt_next = xT, torch.zeros_like(xT).to(device)
                        eps_mu_t_next = custom_ld.accurate_forward(xT, get_model_input_time(ns, T).expand(xT.shape[0]), c=c)
                    
                    ####### Start skip UQ sampling  ######
                    for timestep in range(opt.timesteps//2, 0, -1):

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
                            xt_next, _ , model_s1_var = conditioned_update(ns=ns, x=xt, s=s, t=t, custom_ld=custom_ld, model_s=eps_t, pre_wuq=uq_array[timestep], c=c, r1=0.5)
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
                                    model_t_i, model_t_i_var = custom_ld(xt_next_i, get_model_input_time(ns, s_next).expand(xt_next_i.shape[0]), c=c)
                                    xu_next_i = sample_from_gaussion(torch.exp(log_alpha_s1_next - log_alpha_s_next) * xt_next_i-(sigma_s1_next * phi_11_next) * model_t_i, \
                                                                    torch.square(sigma_s1_next * phi_11_next) * model_t_i_var)
                                    model_u_i, _ = custom_ld(xu_next_i, get_model_input_time(ns, s1_next).expand(xt_next_i.shape[0]), c=c)
                                    list_eps_mu_t_next_i.append(model_u_i)

                                eps_mu_t_next, eps_var_t_next = custom_ld(xt_next, get_model_input_time(ns, s_next).expand(xt_next.shape[0]), c=c)
                                list_xt_next_i = torch.stack(list_xt_next_i, dim=0).to(device)
                                list_eps_mu_t_next_i = torch.stack(list_eps_mu_t_next_i, dim=0).to(device)
                                cov_xt_next_epst_next = torch.mean(list_xt_next_i*list_eps_mu_t_next_i, dim=0)-exp_xt_next*torch.mean(list_eps_mu_t_next_i, dim=0)
                            else:
                                eps_mu_t_next = custom_ld.accurate_forward(xt_next, get_model_input_time(ns, t).expand(xt_next.shape[0]), c=c)

                        else:
                            xt_next, model_s1 = conditioned_update(ns=ns, x=xt, s=s, t=t, custom_ld=custom_ld, model_s=eps_mu_t, pre_wuq=uq_array[timestep], c=c, r1=0.5)
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
                                    model_t_i, model_t_i_var = custom_ld(xt_next_i, get_model_input_time(ns, s_next).expand(xt_next_i.shape[0]), c=c)
                                    xu_next_i = sample_from_gaussion(torch.exp(log_alpha_s1_next - log_alpha_s_next) * xt_next_i-(sigma_s1_next * phi_11_next) * model_t_i, \
                                                                    torch.square(sigma_s1_next * phi_11_next) * model_t_i_var)
                                    model_u_i, _ = custom_ld(xu_next_i, get_model_input_time(ns, s1_next).expand(xt_next_i.shape[0]), c=c)
                                    list_eps_mu_t_next_i.append(model_u_i)

                                eps_mu_t_next, eps_var_t_next = custom_ld(xt_next, get_model_input_time(ns, s_next).expand(xt_next.shape[0]), c=c)
                                list_xt_next_i = torch.stack(list_xt_next_i, dim=0).to(device)
                                list_eps_mu_t_next_i = torch.stack(list_eps_mu_t_next_i, dim=0).to(device)
                                cov_xt_next_epst_next = torch.mean(list_xt_next_i*list_eps_mu_t_next_i, dim=0)-exp_xt_next*torch.mean(list_eps_mu_t_next_i, dim=0)
                            else:
                                eps_mu_t_next = custom_ld.accurate_forward(xt_next, get_model_input_time(ns, t).expand(xt_next.shape[0]), c=c)

                    ####### Save variance and sample image  ######         
                    var_sum[:, loop] = var_xt_next.sum(dim=(1,2,3))
                    x_samples = model.decode_first_stage(xt_next)
                    x = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    # os.makedirs(os.path.join(exp_dir, 'sam/'), exist_ok=True)
                    # for i in range(x.shape[0]):
                    #     path = os.path.join(exp_dir, 'sam/', f"{img_id}.png")
                    #     tvu.save_image(x.cpu()[i].float(), path)
                    #     img_id += 1
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

                print(f'Sampling {total_n_samples} images in {exp_dir}')
                torch.save(var_sum.cpu(), os.path.join(exp_dir, 'var_sum.pt'))

if __name__ == "__main__":
    main()
