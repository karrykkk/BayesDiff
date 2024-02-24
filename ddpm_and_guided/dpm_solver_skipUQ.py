from dpmUQ_utils import exp_iteration, var_iteration, \
inverse_data_transform, sample_from_gaussion, \
NoiseScheduleVP, get_model_input_time, singlestep_dpm_solver_second_update
import torch
import tqdm
import os
import argparse
import yaml
import os
import torch
import numpy as np

from runners.diffusion import Diffusion
from models.diffusion import Model
from models.guided_diffusion.unet import UNetModel as GuidedDiffusion_Model
from models.ema import EMAHelper
from custom_model import CustomModel
from pytorch_lightning import seed_everything
import torch

from la_train_datasets import celeba_dataset, imagenet_dataset
import torchvision.utils as tvu
from torchvision.utils import make_grid
import logging
import time

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

def conditioned_update(ns, x, s, t, custom_model, model_s, pre_wuq, r1=0.5, **model_kwargs):
    if pre_wuq == True:
        return singlestep_dpm_solver_second_update(ns, x, s, t, custom_model, model_s, r1=0.5, **model_kwargs)
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
        model_s1 = custom_model.accurate_forward(x_s1, input_s1.expand(x_s1.shape[0]), **model_kwargs)

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
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach ('generalized'(DDIM) or 'ddpm_noisy'(DDPM) or 'dpmsolver' or 'dpmsolver++')",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="time_uniform",
        help="skip according to ('uniform' or 'quadratic' for DDIM/DDPM; 'logSNR' or 'time_uniform' or 'time_quadratic' for DPM-Solver)",
    )
    parser.add_argument(
        "--base_samples",
        type=str,
        default=None,
        help="base samples for upsampling, *.npz",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--dpm_solver_order", type=int, default=3, help="order of dpm-solver"
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
    parser.add_argument(
        "--dpm_solver_atol", type=float, default=0.0078, help="atol for adaptive step size algorithm"
    )
    parser.add_argument(
        "--dpm_solver_rtol", type=float, default=0.05, help="rtol for adaptive step size algorithm"
    )
    parser.add_argument(
        "--dpm_solver_method",
        type=str,
        default="singlestep",
        help="method of dpm_solver ('adaptive' or 'singlestep' or 'multistep' or 'singlestep_fixed'",
    )
    parser.add_argument(
        "--dpm_solver_type",
        type=str,
        default="dpm_solver",
        help="type of dpm_solver ('dpm_solver' or 'taylor'",
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

    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    args, config = parse_args_and_config()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True

    # set random seed
    seed_everything(args.seed)
    fixed_xT = torch.randn([args.total_n_sample, config.data.channels, config.data.image_size, config.data.image_size])
    total_n_samples = args.total_n_sample
    if total_n_samples % args.sample_batch_size != 0:
        raise ValueError("Total samples for sampling must be divided exactly by args.sample_batch_size, but got {} and {}".format(total_n_samples, args.sample_batch_size))
    n_rounds = total_n_samples // args.sample_batch_size
    if args.fixed_class == 10000:
        fixed_classes = torch.randint(low=0, high=1000, size=(args.sample_batch_size, n_rounds))
    else:
        fixed_classes = torch.randint(low=args.fixed_class, high=args.fixed_class+1, size=(args.sample_batch_size,n_rounds)).to(device)

#########   get la model  ##########  
    diffusion = Diffusion(args, config, rank=device)
    if diffusion.config.model.model_type == "guided_diffusion":
        model = GuidedDiffusion_Model(
            image_size=diffusion.config.model.image_size,
            in_channels=diffusion.config.model.in_channels,
            model_channels=diffusion.config.model.model_channels,
            out_channels=diffusion.config.model.out_channels,
            num_res_blocks=diffusion.config.model.num_res_blocks,
            attention_resolutions=diffusion.config.model.attention_resolutions,
            dropout=diffusion.config.model.dropout,
            channel_mult=diffusion.config.model.channel_mult,
            conv_resample=diffusion.config.model.conv_resample,
            dims=diffusion.config.model.dims,
            num_classes=diffusion.config.model.num_classes,
            use_checkpoint=diffusion.config.model.use_checkpoint,
            use_fp16=diffusion.config.model.use_fp16,
            num_heads=diffusion.config.model.num_heads,
            num_head_channels=diffusion.config.model.num_head_channels,
            num_heads_upsample=diffusion.config.model.num_heads_upsample,
            use_scale_shift_norm=diffusion.config.model.use_scale_shift_norm,
            resblock_updown=diffusion.config.model.resblock_updown,
            use_new_attention_order=diffusion.config.model.use_new_attention_order,
        )
    
    else:
        model = Model(diffusion.config)

    model = model.to(device)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.device}

    if "ckpt_dir" in diffusion.config.model.__dict__.keys():
        ckpt_dir = os.path.expanduser(diffusion.config.model.ckpt_dir)
        states = torch.load(
            ckpt_dir,
            map_location=map_location
        )
        # states = {f"module.{k}":v for k, v in states.items()}
        if diffusion.config.model.model_type == 'improved_ddpm' or diffusion.config.model.model_type == 'guided_diffusion':
            model.load_state_dict(states, strict=True)
            if diffusion.config.model.use_fp16:
                model.convert_to_fp16()
        else:
            modified_states = {}
            for key, value in states[0].items():
                modified_key =  key[7:]
                modified_states[modified_key] = value
            model.load_state_dict(modified_states, strict=True)

        if diffusion.config.model.ema: # for celeba 64x64 in DDIM
            ema_helper = EMAHelper(mu=diffusion.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None
    
    if diffusion.config.data.dataset == "CELEBA":
        train_dataset= celeba_dataset(args = args, config = diffusion.config)
        train_dataloader= torch.utils.data.DataLoader(train_dataset, batch_size=args.train_la_batch_size, shuffle=True)
        custom_model = CustomModel(model, train_dataloader, args, diffusion.config)

    else:
        train_dataset= imagenet_dataset(args = args, config = diffusion.config)
        train_dataloader= torch.utils.data.DataLoader(train_dataset, batch_size=args.train_la_batch_size, shuffle=True)
        custom_model = CustomModel(model, train_dataloader, args, diffusion.config) 
#########   get t sequence (note that t is different from timestep)  ##########
    def get_time_steps(skip_type, t_T, t_0, N, device):
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

    def get_orders_and_timesteps_for_singlestep_solver(steps, t_T, t_0, skip_type, device, order=2):

        if steps % 2 == 0:
            K = steps // 2
            orders = [2,] * K
        else:
            K = steps // 2 + 1
            orders = [2,] * (K - 1) + [1]
        
        if skip_type == 'logSNR':
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0,] + orders), 0).to(device)]

        return timesteps_outer     
    ns = NoiseScheduleVP('discrete', betas=diffusion.betas)    
    t_0 = 1. / ns.total_N
    t_T = ns.T
    t_seq = get_orders_and_timesteps_for_singlestep_solver(steps= args.timesteps, t_T=t_T, t_0=t_0, skip_type=args.skip_type, device=args.device)
    t_seq = torch.flip(t_seq, dims=[0])
    print(t_seq.shape)
#########   get skip UQ rules  ##########  
# if uq_array[i] == False, then we use origin_dpmsolver_update from t_seq[i] to t_seq[i-1]
    uq_array = [False] * (args.timesteps//2+1)
    for i in range(args.timesteps//2, 0, -5):
        uq_array[i] = True
    print(uq_array)

    # fid_dir = f'/data///FID/new_compare/origin/dpm_128/tube_1'
    # fid_dir = f'/data///FID/new_compare/orgin/ddim_128'
    # print(f'uq_array is {uq_array}, fid_dir is {fid_dir}')
#########   start sample  ########## 
    exp_dir = f'/home///dpm_solver_2_exp/skipUQ/{diffusion.config.data.dataset}/{args.fixed_class}_train%{args.train_la_data_size}_step{args.timesteps}_S{args.mc_size}/'
    # os.makedirs(exp_dir, exist_ok=True)
    total_n_samples = args.total_n_sample
    if total_n_samples % args.sample_batch_size != 0:
        raise ValueError("Total samples for sampling must be divided exactly by args.sample_batch_size, but got {} and {}".format(total_n_samples, args.sample_batch_size))
    n_rounds = total_n_samples // args.sample_batch_size
    var_sum = torch.zeros((args.sample_batch_size, n_rounds)).to(device)
    sample_x =[]
    img_id = 1000000
    with torch.no_grad():
        for loop in tqdm.tqdm(
            range(n_rounds), desc="Generating image samples for FID evaluation."
        ):
            
            if diffusion.config.sampling.cond_class:
                classes = fixed_classes[:, loop].to(device)
            else:
                classes = None

            if classes is None:
                model_kwargs = {}
            else:
                model_kwargs = {"y": classes}

            xT = fixed_xT[loop*args.sample_batch_size:(loop+1)*args.sample_batch_size, :, :, :].to(device)               
            timestep, mc_sample_size  = args.timesteps//2, args.mc_size

            ###### Initialize
            T = t_seq[timestep]
            if uq_array[timestep] == True:
                xt_next = xT
                exp_xt_next, var_xt_next = xT, torch.zeros_like(xT).to(device)
                eps_mu_t_next, eps_var_t_next = custom_model(xT, get_model_input_time(ns, T).expand(xT.shape[0]), **model_kwargs) 
                cov_xt_next_epst_next = torch.zeros_like(xT).to(device)
                _, model_s1, _ = conditioned_update(ns, xt_next, T, t_seq[timestep-1], custom_model, eps_mu_t_next, pre_wuq=True, r1=0.5, **model_kwargs)
                list_eps_mu_t_next_i = torch.unsqueeze(model_s1, dim=0)
            else:
                xt_next = xT
                exp_xt_next, var_xt_next = xT, torch.zeros_like(xT).to(device)
                eps_mu_t_next = custom_model.accurate_forward(xT, get_model_input_time(ns, T).expand(xT.shape[0]), **model_kwargs)
            
            ####### Start skip UQ sampling
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
                    xt_next, _ , model_s1_var = conditioned_update(ns=ns, x=xt, s=s, t=t, custom_model=custom_model, model_s=eps_t, pre_wuq=uq_array[timestep], r1=0.5, **model_kwargs)
                    exp_xt_next = conditioned_exp_iteration(exp_xt, ns, s, t, pre_wuq=uq_array[timestep], mc_eps_exp_s1=mc_eps_exp_t)
                    var_xt_next = conditioned_var_iteration(var_xt, ns, s, t, pre_wuq=uq_array[timestep], cov_xt_epst= cov_xt_epst, var_epst=model_s1_var)
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
                            model_t_i, model_t_i_var = custom_model(xt_next_i, get_model_input_time(ns, s_next).expand(xt_next_i.shape[0]), **model_kwargs)
                            xu_next_i = sample_from_gaussion(torch.exp(log_alpha_s1_next - log_alpha_s_next) * xt_next_i-(sigma_s1_next * phi_11_next) * model_t_i, \
                                                            torch.square(sigma_s1_next * phi_11_next) * model_t_i_var)
                            model_u_i, _ = custom_model(xu_next_i, get_model_input_time(ns, s1_next).expand(xt_next_i.shape[0]), **model_kwargs)
                            list_eps_mu_t_next_i.append(model_u_i)

                        eps_mu_t_next, eps_var_t_next = custom_model(xt_next, get_model_input_time(ns, s_next).expand(xt_next.shape[0]), **model_kwargs)
                        list_xt_next_i = torch.stack(list_xt_next_i, dim=0).to(device)
                        list_eps_mu_t_next_i = torch.stack(list_eps_mu_t_next_i, dim=0).to(device)
                        cov_xt_next_epst_next = torch.mean(list_xt_next_i*list_eps_mu_t_next_i, dim=0)-exp_xt_next*torch.mean(list_eps_mu_t_next_i, dim=0)
                    else:
                        eps_mu_t_next = custom_model.accurate_forward(xt_next, get_model_input_time(ns, t).expand(xt_next.shape[0]), **model_kwargs)

                else:
                    xt_next, model_s1 = conditioned_update(ns=ns, x=xt, s=s, t=t, custom_model=custom_model, model_s=eps_mu_t, pre_wuq=uq_array[timestep], r1=0.5, **model_kwargs)
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
                            model_t_i, model_t_i_var = custom_model(xt_next_i, get_model_input_time(ns, s_next).expand(xt_next_i.shape[0]), **model_kwargs)
                            xu_next_i = sample_from_gaussion(torch.exp(log_alpha_s1_next - log_alpha_s_next) * xt_next_i-(sigma_s1_next * phi_11_next) * model_t_i, \
                                                            torch.square(sigma_s1_next * phi_11_next) * model_t_i_var)
                            model_u_i, _ = custom_model(xu_next_i, get_model_input_time(ns, s1_next).expand(xt_next_i.shape[0]), **model_kwargs)
                            list_eps_mu_t_next_i.append(model_u_i)

                        eps_mu_t_next, eps_var_t_next = custom_model(xt_next, get_model_input_time(ns, s_next).expand(xt_next.shape[0]), **model_kwargs)
                        list_xt_next_i = torch.stack(list_xt_next_i, dim=0).to(device)
                        list_eps_mu_t_next_i = torch.stack(list_eps_mu_t_next_i, dim=0).to(device)
                        cov_xt_next_epst_next = torch.mean(list_xt_next_i*list_eps_mu_t_next_i, dim=0)-exp_xt_next*torch.mean(list_eps_mu_t_next_i, dim=0)
                    else:
                        eps_mu_t_next = custom_model.accurate_forward(xt_next, get_model_input_time(ns, t).expand(xt_next.shape[0]), **model_kwargs)
        
                # os.makedirs(f'{exp_dir}trajectory/sam', exist_ok=True)
                # os.makedirs(f'{exp_dir}trajectory/exp', exist_ok=True)
                # os.makedirs(f'{exp_dir}trajectory/dev', exist_ok=True)
                # for i in range(args.sample_batch_size):
                #     tvu.save_image(inverse_data_transform(diffusion.config, exp_xt_next[i]), f'{exp_dir}trajectory/exp/timestep_{timestep}_expectation_{i+loop*args.sample_batch_size}.png')
                #     tvu.save_image(torch.sqrt(var_xt_next[i]), f'{exp_dir}trajectory/dev/timestep_{timestep}_deviation_{i+loop*args.sample_batch_size}.png')
                #     tvu.save_image(inverse_data_transform(diffusion.config, xt_next[i]), f'{exp_dir}trajectory/sam/timestep_{timestep}_sample_{i+loop*args.sample_batch_size}.png')
                
            var_sum[:, loop] = var_xt_next.sum(dim=(1,2,3))    
            x = inverse_data_transform(config, xt_next)
            sample_x.append(x)         
            os.makedirs(os.path.join(exp_dir, 'sam/'), exist_ok=True)
            # os.makedirs(os.path.join(fid_dir, 'var/'), exist_ok=True)
            for i in range(x.shape[0]):
                path = os.path.join(exp_dir, 'sam/', f"{img_id}.png")
                # var_path = os.path.join(fid_dir, 'var/', f"{img_id}.pt")
                tvu.save_image(x[i], path)
                # torch.save(var_xt_next.sum(dim=(1,2,3))[i].cpu(), var_path)
                img_id += 1

        sample_x = torch.concat(sample_x, dim=0)
        var = []
        for j in range(n_rounds):
            var.append(var_sum[:, j])
        var = torch.concat(var, dim=0)
        sorted_var, sorted_indices = torch.sort(var, descending=True)
        reordered_sample_x = torch.index_select(sample_x, dim=0, index=sorted_indices.int())
        grid_sample_x = make_grid(reordered_sample_x, nrow=12, padding=1)
        tvu.save_image(grid_sample_x.cpu().float(), os.path.join(exp_dir, "sorted_sample.png"))

        # print(f'Sampling {total_n_samples} images in {fid_dir}')
        # torch.save(var_sum.cpu(), os.path.join(fid_dir, 'var_sum.pt'))

if __name__ == "__main__":
    main()

    


    



