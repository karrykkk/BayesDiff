import argparse, os
import torch
import math
from itertools import islice
from omegaconf import OmegaConf
from tqdm import tqdm
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from custom_ld import CustomLD
from dataset import laion_dataset
import torchvision.utils as tvu
from ldm.util import instantiate_from_config
from utils import NoiseScheduleVP, get_model_input_time
from ddimUQ_utils import compute_alpha, singlestep_ddim_sample, var_iteration, exp_iteration, \
    sample_from_gaussion

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

def conditioned_exp_iteration(model, exp_xt, seq, timestep, pre_wuq, mc_eps_exp_t=None, acc_eps_t = None):
    if pre_wuq == True:
        return exp_iteration(model, exp_xt, seq, timestep, mc_eps_exp_t)
    else:
        return exp_iteration(model, exp_xt, seq, timestep, acc_eps_t)
    
def conditioned_var_iteration(model, var_xt, cov_xt_epst, var_epst, seq, timestep, pre_wuq):

    if pre_wuq == True:
        return var_iteration(model, var_xt, cov_xt_epst, var_epst, seq, timestep)
    else:
        n = var_xt.size(0)
        t = (torch.ones(n)*seq[timestep]).to(var_xt.device)
        next_t = (torch.ones(n)*seq[(timestep-1)]).to(var_xt.device)
        at = compute_alpha(model.betas, t.long())
        at_next = compute_alpha(model.betas, next_t.long())
        var_xt_next = (at_next/at) * var_xt

        return var_xt_next

def get_scaled_var_eps(scale, var_eps_c, var_eps_uc):
    return pow(1-scale, 2)* var_eps_uc + pow(scale, 2)* var_eps_c
def get_scaled_exp_eps(scale, exp_eps_c, exp_eps_uc):
    return (1-scale)* exp_eps_uc + scale* exp_eps_c

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())
 
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a cheetah drinking an espresso",
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
        default=7.5,
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
        default=1234,
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
        "--from_file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument("--mc_size", type=int, default=10)
    parser.add_argument("--sample_batch_size", type=int, default=8)
    parser.add_argument("--train_la_batch_size", type=int, default=4)
    parser.add_argument("--train_la_data_size", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default= 50)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--total_n_samples', type=int, default=80)
    parser.add_argument('--cut', type=int, default=40)
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

    fixed_xT = torch.randn([opt.sample_batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
##########   get t sequence (note that t is different from timestep)  ########## 

    skip = model.num_timesteps // opt.timesteps
    seq = range(0, model.num_timesteps, skip)

#########   get skip UQ rules  ##########  
# if uq_array[i] == False, then we use origin_dpmsolver_update from t_seq[i] to t_seq[i-1]
    uq_array = [False] * (opt.timesteps)
    cut = opt.cut
    for i in range(opt.timesteps-1, cut, -2):
        uq_array[i] = True
    
#########   get prompt  ##########  
    if opt.from_file:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
    else:
        c = model.get_learned_conditioning(opt.prompt)
        c = torch.concat(opt.sample_batch_size * [c], dim=0)
        uc = model.get_learned_conditioning(opt.sample_batch_size * [""])
        exp_dir = f'/home///ddim_skipUQ/var_use'
        os.makedirs(exp_dir, exist_ok=True)

#########   start sample  ########## 
    
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for prompts in tqdm(data):
                    for j in range(opt.sample_batch_size):
                        img_id = 1000000
                        c = model.get_learned_conditioning(prompts)
                        c = torch.concat(1 * [c], dim=0)
                        uc = model.get_learned_conditioning(1 * [""])
                        exp_dir = f'/home///ddim_skipUQ/paper/prior_precison{1}_train_data_size{opt.train_la_data_size}_cut{opt.cut}_{prompts}'
                        os.makedirs(exp_dir, exist_ok=True)

                        xT, timestep, mc_sample_size  = fixed_xT[j, :, :, :], opt.timesteps-1, opt.mc_size
                        xT = torch.unsqueeze(xT, dim=0)
                        T = seq[timestep]
                        if uq_array[timestep] == True:
                            xt_next = xT
                            exp_xt_next, var_xt_next = xT, torch.zeros_like(xT).to(device)
                            eps_mu_t_next_c, eps_var_t_next_c = custom_ld(xT, (torch.ones(1) * T).to(xT.device), c=c) 
                            eps_mu_t_next_uc, eps_var_t_next_uc = custom_ld(xT, (torch.ones(1) * T).to(xT.device), c=uc) 
                            eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=eps_mu_t_next_c, exp_eps_uc=eps_mu_t_next_uc)
                            eps_var_t_next = get_scaled_var_eps(opt.scale, eps_var_t_next_c, eps_var_t_next_uc)
                            cov_xt_next_epst_next = torch.zeros_like(xT).to(device)
                            list_eps_mu_t_next_i = torch.unsqueeze(eps_mu_t_next, dim=0)
                        else:
                            xt_next = xT
                            exp_xt_next, var_xt_next = xT, torch.zeros_like(xT).to(device)
                            eps_mu_t_next_c = custom_ld.accurate_forward(xT, (torch.ones(1) * T).to(xT.device), c=c)
                            eps_mu_t_next_uc = custom_ld.accurate_forward(xT, (torch.ones(1) * T).to(xT.device), c=uc)
                            eps_mu_t_next = get_scaled_exp_eps(opt.scale, eps_mu_t_next_c, eps_mu_t_next_uc)

                        for timestep in range(opt.timesteps-1, cut, -1):

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
                                eps_t= sample_from_gaussion(eps_mu_t, eps_var_t)
                                xt_next = singlestep_ddim_sample(model, xt, seq, timestep, eps_t)
                                exp_xt_next = conditioned_exp_iteration(model, exp_xt, seq, timestep, pre_wuq=uq_array[timestep], mc_eps_exp_t=mc_eps_exp_t)
                                var_xt_next = conditioned_var_iteration(model, var_xt, cov_xt_epst, var_epst=eps_var_t, seq=seq, timestep=timestep, pre_wuq= uq_array[timestep])
                                if uq_array[timestep-1] == True:
                                    list_xt_next_i, list_eps_mu_t_next_i=[], []
                                    for _ in range(mc_sample_size):
                                        var_xt_next = torch.clamp(var_xt_next, min=0)
                                        xt_next_i = sample_from_gaussion(exp_xt_next, var_xt_next)
                                        list_xt_next_i.append(xt_next_i)
                                        eps_mu_t_next_i_c, _ = custom_ld(xt_next_i, (torch.ones(1) * seq[timestep-1]).to(xt.device), c=c)
                                        eps_mu_t_next_i_uc, _ = custom_ld(xt_next_i, (torch.ones(1) * seq[timestep-1]).to(xt.device), c=uc)
                                        eps_mu_t_next_i = get_scaled_exp_eps(opt.scale, eps_mu_t_next_i_c, eps_mu_t_next_i_uc)
                                        list_eps_mu_t_next_i.append(eps_mu_t_next_i)
                                            
                                    eps_mu_t_next_c, eps_var_t_next_c = custom_ld(xt_next, (torch.ones(1) * seq[timestep-1]).to(xt.device), c=c)
                                    eps_mu_t_next_uc, eps_var_t_next_uc = custom_ld(xt_next, (torch.ones(1) * seq[timestep-1]).to(xt.device), c=uc)
                                    eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=eps_mu_t_next_c, exp_eps_uc=eps_mu_t_next_uc)
                                    eps_var_t_next = get_scaled_var_eps(opt.scale, eps_var_t_next_c, eps_var_t_next_uc)
                                    list_xt_next_i = torch.stack(list_xt_next_i, dim=0).to(device)
                                    list_eps_mu_t_next_i = torch.stack(list_eps_mu_t_next_i, dim=0).to(device)
                                    cov_xt_next_epst_next = torch.mean(list_xt_next_i*list_eps_mu_t_next_i, dim=0)-exp_xt_next*torch.mean(list_eps_mu_t_next_i, dim=0)
                                else:
                                    eps_mu_t_next_c = custom_ld.accurate_forward(xt_next, (torch.ones(1) * seq[timestep-1]).to(xt.device), c=c)
                                    eps_mu_t_next_uc = custom_ld.accurate_forward(xt_next, (torch.ones(1) * seq[timestep-1]).to(xt.device), c=uc)
                                    eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=eps_mu_t_next_c, exp_eps_uc=eps_mu_t_next_uc)
                            else:
                                xt_next = singlestep_ddim_sample(model, xt, seq, timestep, eps_mu_t)
                                exp_xt_next = conditioned_exp_iteration(model, exp_xt, seq, timestep, pre_wuq=uq_array[timestep], acc_eps_t = eps_mu_t)
                                var_xt_next = conditioned_var_iteration(model, var_xt, cov_xt_epst= None, var_epst=None, seq= seq, timestep=timestep, pre_wuq= uq_array[timestep])
                                if uq_array[timestep-1] == True:
                                    list_xt_next_i, list_eps_mu_t_next_i=[], []
                                    for _ in range(mc_sample_size):
                                        var_xt_next = torch.clamp(var_xt_next, min=0)
                                        xt_next_i = sample_from_gaussion(exp_xt_next, var_xt_next)
                                        list_xt_next_i.append(xt_next_i)
                                        eps_mu_t_next_i_c, _ = custom_ld(xt_next_i, (torch.ones(1) * seq[timestep-1]).to(xt.device), c=c)
                                        eps_mu_t_next_i_uc, _ = custom_ld(xt_next_i, (torch.ones(1) * seq[timestep-1]).to(xt.device), c=uc)
                                        eps_mu_t_next_i = get_scaled_exp_eps(opt.scale, eps_mu_t_next_i_c, eps_mu_t_next_i_uc)
                                        list_eps_mu_t_next_i.append(eps_mu_t_next_i)
                                            
                                    eps_mu_t_next_c, eps_var_t_next_c = custom_ld(xt_next, (torch.ones(1) * seq[timestep-1]).to(xt.device), c=c)
                                    eps_mu_t_next_uc, eps_var_t_next_uc = custom_ld(xt_next, (torch.ones(1) * seq[timestep-1]).to(xt.device), c=uc)
                                    eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=eps_mu_t_next_c, exp_eps_uc=eps_mu_t_next_uc)
                                    eps_var_t_next = get_scaled_var_eps(opt.scale, eps_var_t_next_c, eps_var_t_next_uc)
                                    list_xt_next_i = torch.stack(list_xt_next_i, dim=0).to(device)
                                    list_eps_mu_t_next_i = torch.stack(list_eps_mu_t_next_i, dim=0).to(device)
                                    cov_xt_next_epst_next = torch.mean(list_xt_next_i*list_eps_mu_t_next_i, dim=0)-exp_xt_next*torch.mean(list_eps_mu_t_next_i, dim=0)
                                else:
                                    eps_mu_t_next_c = custom_ld.accurate_forward(xt_next, (torch.ones(1) * seq[timestep-1]).to(xt.device), c=c)
                                    eps_mu_t_next_uc = custom_ld.accurate_forward(xt_next, (torch.ones(1) * seq[timestep-1]).to(xt.device), c=uc)
                                    eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=eps_mu_t_next_c, exp_eps_uc=eps_mu_t_next_uc)
                        
                        diversity_sample = []
                        for loop in range(6):
                            diversity = 7
                            new_xt = []
                            c = model.get_learned_conditioning(prompts)
                            c = torch.concat((diversity+1) * [c], dim=0)
                            uc = model.get_learned_conditioning((diversity+1) * [""])
                            var_xt_next = torch.clamp(var_xt_next, min=0)
                            print(var_xt_next.sum())
                            new_xt.append(sample_from_gaussion(exp_xt_next, torch.zeros_like(exp_xt_next).to(device)))
                            for i in range(diversity):
                                new_xt.append(sample_from_gaussion(exp_xt_next, var_xt_next))
                            new_xt = torch.concat(new_xt, dim=0)
                            print(new_xt.shape)
                            new_eps_mu_t_next_c = custom_ld.accurate_forward(new_xt, (torch.ones((diversity+1)) * seq[cut]).to(new_xt.device), c=c)
                            new_eps_mu_t_next_uc = custom_ld.accurate_forward(new_xt, (torch.ones((diversity+1)) * seq[cut]).to(new_xt.device), c=uc)
                            new_eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=new_eps_mu_t_next_c, exp_eps_uc=new_eps_mu_t_next_uc)
                            for timestep in range(cut, 0, -1):
                                if timestep == cut:
                                    xt = new_xt
                                    eps_mu_t = new_eps_mu_t_next
                                else:
                                    xt = xt_next
                                    eps_mu_t = eps_mu_t_next     
                                xt_next = singlestep_ddim_sample(model, xt, seq, timestep, eps_mu_t)
                                eps_mu_t_next_c = custom_ld.accurate_forward(xt_next, (torch.ones((diversity+1)) * seq[timestep-1]).to(xt.device), c=c)
                                eps_mu_t_next_uc = custom_ld.accurate_forward(xt_next, (torch.ones((diversity+1)) * seq[timestep-1]).to(xt.device), c=uc)
                                eps_mu_t_next = get_scaled_exp_eps(scale=opt.scale, exp_eps_c=eps_mu_t_next_c, exp_eps_uc=eps_mu_t_next_uc)

                            x_samples = model.decode_first_stage(xt_next)
                            x = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                            diversity_sample.append(torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0))
                            os.makedirs(os.path.join(exp_dir, f'{j}'), exist_ok=True)
                            for i in range(diversity):
                                path = os.path.join(exp_dir, f'{j}', f"{img_id}.png")
                                tvu.save_image(x.cpu()[i].float(), path)
                                img_id += 1
                        
                        diversity_sample = torch.concat(diversity_sample, dim=0)
                        diversity_sample = tvu.make_grid(diversity_sample, nrow=8, padding=2)
                        tvu.save_image(diversity_sample.cpu().float(), os.path.join(exp_dir, f'{j}', f"diversity_sample_{j}.png"))

if __name__ == "__main__":
    main()