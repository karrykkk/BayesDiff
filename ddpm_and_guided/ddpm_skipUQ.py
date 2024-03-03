import argparse
import yaml
import os
import torch
import numpy as np

from runners.diffusion import Diffusion
from models.diffusion import Model
from models.improved_ddpm.unet import UNetModel as ImprovedDDPM_Model
from models.guided_diffusion.unet import UNetModel as GuidedDiffusion_Model
from models.ema import EMAHelper
from custom_model import CustomModel
from pytorch_lightning import seed_everything
import torch
import warnings

from la_train_datasets import celeba_dataset, imagenet_dataset
import torchvision.utils as tvu
from torchvision.utils import make_grid
import logging
import copy
import tqdm
import time


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)

def compute_alpha(beta, t):
    # beta is the \beta in ddpm paper
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def singlestep_ddpm_sample(diffusion,xt,seq,timestep,eps_t):
    #at.sqrt() is the \alpha_t in our paper
    n = xt.size(0)
    t = (torch.ones(n)*seq[timestep]).to(xt.device)
    next_t = (torch.ones(n)*seq[timestep-1]).to(xt.device)
    at = compute_alpha(diffusion.betas,t.long())
    at_minus_1 = compute_alpha(diffusion.betas,next_t.long())
    beta_t = 1 - at/at_minus_1
    
    mean = (1/(1-beta_t).sqrt())*(xt - beta_t * eps_t / ( 1 - at ).sqrt())
    
    noise = torch.randn_like(xt)
    logvar = beta_t.log()
    xt_next = mean + torch.exp(logvar * 0.5) * noise
    
    return xt_next
    
def ddpm_exp_iteration(diffusion,exp_xt,seq,timestep,mc_eps_exp_t):
    # at here is the \bar{\alpha}_t in ddpm paper
    n = exp_xt.size(0)
    t = (torch.ones(n)*seq[timestep]).to(exp_xt.device)
    next_t = (torch.ones(n)*seq[timestep-1]).to(exp_xt.device)
    at = compute_alpha(diffusion.betas,t.long())
    at_minus_1 = compute_alpha(diffusion.betas,t.long())
    beta_t = 1 - at / at_minus_1
    exp_eps_coefficient = -1 * beta_t / ((1 - beta_t) * (1 - at) ).sqrt()
    exp_xt_next = (1 / (1 - beta_t).sqrt() ) * exp_xt + exp_eps_coefficient * mc_eps_exp_t
    return exp_xt_next
       
def ddpm_var_iteration(diffusion, var_xt, cov_xt_epst, var_epst, seq, timestep):
    # at is the \bar{\alpha}_t in ddpm paper
    n = var_xt.size(0)
    t = (torch.ones(n)*seq[timestep]).to(var_xt.device)
    next_t = (torch.ones(n)*seq[timestep-1]).to(var_xt.device)
    at = compute_alpha(diffusion.betas,t.long())
    at_minus_1 = compute_alpha(diffusion.betas,next_t.long())
    beta_t = 1 - at/at_minus_1
    cov_coefficient = (-2 * beta_t) / ( (1 - beta_t) * (1 - at).sqrt() )
    var_epst_coefficient = (beta_t ** 2) / ((1 - beta_t) * (1 - at))
    var_xt_next = (1 / (1 - beta_t).sqrt()) * var_xt + cov_coefficient * cov_xt_epst + var_epst_coefficient * var_epst + beta_t
    
    return var_xt_next

def conditioned_exp_iteration(diffusion, exp_xt, seq, timestep, pre_wuq, mc_eps_exp_t=None, acc_eps_t = None):
    if pre_wuq == True:
        return ddpm_exp_iteration(diffusion, exp_xt, seq, timestep, mc_eps_exp_t)
    else:
        return ddpm_exp_iteration(diffusion, exp_xt, seq, timestep, acc_eps_t)

def conditioned_var_iteration(diffusion, var_xt, cov_xt_epst, var_epst, seq, timestep, pre_wuq):

    if pre_wuq == True:
        return ddpm_var_iteration(diffusion, var_xt, cov_xt_epst, var_epst, seq, timestep)
    else:
        # at is the \bar{\alpha}_t in ddpm paper
        n = var_xt.size(0)
        t = (torch.ones(n)*seq[timestep]).to(var_xt.device)
        next_t = (torch.ones(n)*seq[timestep-1]).to(var_xt.device)
        at = compute_alpha(diffusion.betas,t.long())
        at_minus_1 = compute_alpha(diffusion.betas,next_t.long())
        beta_t = 1 - at/at_minus_1
        var_xt_next = (1 / (1 - beta_t).sqrt()) * var_xt + beta_t
        
        return var_xt_next

def sample_from_gaussian(mean, var):
    samples = mean + (torch.randn(mean.shape).to(mean.device)) * torch.sqrt(var)    
    return samples

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
        "--timesteps", type=int, default=250, help="number of steps involved"
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
    args , config = parse_args_and_config()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True
    

    # set random seed
    seed_everything(args.seed)
    fixed_xT = torch.randn([args.total_n_sample, config.data.channels, config.data.image_size, config.data.image_size], device=device) 
    
    # initialize diffusion and model(unet)
    diffusion = Diffusion(args, config, rank=device)
    if diffusion.config.model.model_type == 'improved_ddpm':
        model = ImprovedDDPM_Model(
            in_channels=diffusion.config.model.in_channels,
            model_channels=diffusion.config.model.model_channels,
            out_channels=diffusion.config.model.out_channels,
            num_res_blocks=diffusion.config.model.num_res_blocks,
            attention_resolutions=diffusion.config.model.attention_resolutions,
            dropout=diffusion.config.model.dropout,
            channel_mult=diffusion.config.model.channel_mult,
            conv_resample=diffusion.config.model.conv_resample,
            dims=diffusion.config.model.dims,
            use_checkpoint=diffusion.config.model.use_checkpoint,
            num_heads=diffusion.config.model.num_heads,
            num_heads_upsample=diffusion.config.model.num_heads_upsample,
            use_scale_shift_norm=diffusion.config.model.use_scale_shift_norm
        )
    elif diffusion.config.model.model_type == "guided_diffusion":
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

    ##########   get t sequence (note that t is different from timestep)  ########## 

    if diffusion.args.skip_type == "uniform":
        skip = diffusion.num_timesteps // diffusion.args.timesteps
        seq = range(0, diffusion.num_timesteps, skip)
    elif diffusion.args.skip_type == "quad":
        seq = (
            np.linspace(
                0, np.sqrt(diffusion.num_timesteps * 0.8), diffusion.args.timesteps
            )
            ** 2
        )
        seq = [int(s) for s in list(seq)]
    else:
        raise NotImplementedError   
#########   get skip UQ rules  ##########  
    # if uq_array[i] == False, then we use origin_dpmsolver_update from t_seq[i] to t_seq[i-1]
    uq_array = [False] * (args.timesteps)
    for i in range(args.timesteps-1, 0, -5):
        uq_array[i] = True
    
    total_n_samples = args.total_n_sample
    sample_x = []
    if total_n_samples % args.sample_batch_size != 0:
        raise ValueError("Total samples for sampling must be divided exactly by args.sample_batch_size, but got {} and {}".format(total_n_samples, args.sample_batch_size))
    n_rounds = total_n_samples // args.sample_batch_size
    var_sum = torch.zeros((args.sample_batch_size, n_rounds)).to(device)
    img_id = 1000000
    samle_batch_size = args.sample_batch_size
    with torch.no_grad():
        for loop in tqdm.tqdm(
            range(n_rounds), desc="Generating image samples for FID evaluation."
        ):
            
            if diffusion.config.sampling.cond_class:
                if diffusion.args.fixed_class == 10000:
                    classes = torch.randint(low=0, high=diffusion.config.data.num_classes, size=(args.sample_batch_size,)).to(device)
                else:
                    classes = torch.randint(low=diffusion.args.fixed_class, high=diffusion.args.fixed_class + 1, size=(args.sample_batch_size,)).to(device)
            else:
                classes = None

            if classes is None:
                model_kwargs = {}
            else:
                model_kwargs = {"y": classes}
      
            samle_batch_size = args.sample_batch_size
            xT = fixed_xT[loop*args.sample_batch_size:(loop+1)*args.sample_batch_size, :, :, :]
            timestep, mc_sample_size = args.timesteps-1, args.mc_size
            T = (torch.ones(samle_batch_size) * seq[timestep]).to(xT.device)
            if uq_array[timestep] == True:
                xt_next = xT
                exp_xt_next, var_xt_next = xT, torch.zeros_like(xT).to(device)
                eps_mu_t_next, eps_var_t_next = custom_model(xT, T, **model_kwargs) 
                cov_xt_next_epst_next = torch.zeros_like(xT).to(device)
                list_eps_mu_t_next_i = torch.unsqueeze(eps_mu_t_next, dim=0)
            else:
                xt_next = xT
                exp_xt_next, var_xt_next = xT, torch.zeros_like(xT).to(device)
                eps_mu_t_next = custom_model.accurate_forward(xT, (torch.ones(samle_batch_size) * seq[timestep]).to(xT.device), **model_kwargs)
    
            for timestep in range(args.timesteps-1, 0, -1):

                if uq_array[timestep] == True:
                    xt = xt_next
                    exp_xt, var_xt = exp_xt_next, var_xt_next
                    eps_mu_t, eps_var_t, cov_xt_epst = eps_mu_t_next, eps_var_t_next, cov_xt_next_epst_next
                    mc_eps_exp_t = torch.mean(list_eps_mu_t_next_i, dim=0)
                else: 
                    xt = xt_next
                    exp_xt, var_xt = exp_xt_next, var_xt_next
                    eps_mu_t = eps_mu_t_next

                if uq_array[timestep] == True:
                    eps_t= sample_from_gaussian(eps_mu_t, eps_var_t)
                    xt_next = singlestep_ddpm_sample(diffusion, xt, seq, timestep, eps_t)
                    exp_xt_next = conditioned_exp_iteration(diffusion, exp_xt, seq, timestep, pre_wuq=uq_array[timestep], mc_eps_exp_t=mc_eps_exp_t)
                    var_xt_next = conditioned_var_iteration(diffusion, var_xt, cov_xt_epst, var_epst=eps_var_t, seq=seq, timestep=timestep, pre_wuq= uq_array[timestep])
                    if uq_array[timestep-1] == True:
                        list_xt_next_i, list_eps_mu_t_next_i=[], []
                        for _ in range(mc_sample_size):
                            var_xt_next = torch.clamp(var_xt_next, min=0)
                            xt_next_i = sample_from_gaussian(exp_xt_next, var_xt_next)
                            list_xt_next_i.append(xt_next_i)
                            eps_mu_t_next_i, _ = custom_model(xt_next_i, (torch.ones(samle_batch_size) * seq[timestep-1]).to(xt.device), **model_kwargs)
                            list_eps_mu_t_next_i.append(eps_mu_t_next_i)
                                
                        eps_mu_t_next, eps_var_t_next = custom_model(xt_next, (torch.ones(samle_batch_size) * seq[timestep-1]).to(xt.device), **model_kwargs)
                        list_xt_next_i = torch.stack(list_xt_next_i, dim=0).to(device)
                        list_eps_mu_t_next_i = torch.stack(list_eps_mu_t_next_i, dim=0).to(device)
                        cov_xt_next_epst_next = torch.mean(list_xt_next_i*list_eps_mu_t_next_i, dim=0)-exp_xt_next*torch.mean(list_eps_mu_t_next_i, dim=0)
                    else:
                        eps_mu_t_next = custom_model.accurate_forward(xt_next, (torch.ones(samle_batch_size) * seq[timestep-1]).to(xt.device), **model_kwargs)
                else:
                    xt_next = singlestep_ddpm_sample(diffusion, xt, seq, timestep, eps_mu_t)
                    exp_xt_next = conditioned_exp_iteration(diffusion, exp_xt, seq, timestep, pre_wuq=uq_array[timestep], acc_eps_t = eps_mu_t)
                    var_xt_next = conditioned_var_iteration(diffusion, var_xt, cov_xt_epst= None, var_epst=None, seq= seq, timestep=timestep, pre_wuq= uq_array[timestep])
                    if uq_array[timestep-1] == True:
                        list_xt_next_i, list_eps_mu_t_next_i=[], []
                        for _ in range(mc_sample_size):
                            var_xt_next = torch.clamp(var_xt_next, min=0)
                            xt_next_i = sample_from_gaussian(exp_xt_next, var_xt_next)
                            list_xt_next_i.append(xt_next_i)
                            eps_mu_t_next_i, _ = custom_model(xt_next_i, (torch.ones(samle_batch_size) * seq[timestep-1]).to(xt.device), **model_kwargs)
                            list_eps_mu_t_next_i.append(eps_mu_t_next_i)
                                
                        eps_mu_t_next, eps_var_t_next = custom_model(xt_next, (torch.ones(samle_batch_size) * seq[timestep-1]).to(xt.device), **model_kwargs)
                        list_xt_next_i = torch.stack(list_xt_next_i, dim=0).to(device)
                        list_eps_mu_t_next_i = torch.stack(list_eps_mu_t_next_i, dim=0).to(device)
                        cov_xt_next_epst_next = torch.mean(list_xt_next_i*list_eps_mu_t_next_i, dim=0)-exp_xt_next*torch.mean(list_eps_mu_t_next_i, dim=0)
                    else:
                        eps_mu_t_next = custom_model.accurate_forward(xt_next, (torch.ones(samle_batch_size) * seq[timestep-1]).to(xt.device), **model_kwargs)

            var_sum[:, loop] = var_xt_next.sum(dim=(1,2,3))    
            x = inverse_data_transform(config, xt_next)
            sample_x.append(x)
            
        exp_dir = f'./exp/{diffusion.config.data.dataset}/ddpm_exp_fixed_class{args.fixed_class}_train%{args.train_la_data_size}_step{args.timesteps}_S{args.mc_size}/'
        os.makedirs(exp_dir, exist_ok=True)
        sample_x = torch.concat(sample_x, dim=0)
        var = []       
        for j in range(n_rounds):
            var.append(var_sum[:, j])
        var = torch.concat(var, dim=0)
        sorted_var, sorted_indices = torch.sort(var, descending=True)
        reordered_sample_x = torch.index_select(sample_x, dim=0, index=sorted_indices.int())
        grid_sample_x = make_grid(reordered_sample_x, nrow=8, padding=2)
        tvu.save_image(grid_sample_x.cpu().float(), os.path.join(exp_dir, "sorted_sample.png"))

if __name__ == "__main__":
    main()
