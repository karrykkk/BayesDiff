import numpy as np
from la_train_datasets import celeba_dataset, imagenet_dataset
from pytorch_lightning import seed_everything
import os
import torch
from runners.diffusion import Diffusion
from models.diffusion import Model, flattened_Model
from models.improved_ddpm.unet import UNetModel as ImprovedDDPM_Model
from models.guided_diffusion.unet import UNetModel as GuidedDiffusion_Model
from models.ema import EMAHelper
from custom_model import CustomModel
import torchvision.utils as tvu
import logging
import time
import tqdm
from ddimUQ_utils import inverse_data_transform, compute_alpha, singlestep_ddim_sample, \
var_iteration, exp_iteration, sample_from_gaussion, parse_args_and_config

def conditioned_exp_iteration(diffusion, exp_xt, seq, timestep, pre_wuq, mc_eps_exp_t=None, acc_eps_t = None):
    if pre_wuq == True:
        return exp_iteration(diffusion, exp_xt, seq, timestep, mc_eps_exp_t)
    else:
        return exp_iteration(diffusion, exp_xt, seq, timestep, acc_eps_t)

def conditioned_var_iteration(diffusion, var_xt, cov_xt_epst, var_epst, seq, timestep, pre_wuq):

    if pre_wuq == True:
        return var_iteration(diffusion, var_xt, cov_xt_epst, var_epst, seq, timestep)
    else:
        n = var_xt.size(0)
        t = (torch.ones(n)*seq[timestep]).to(var_xt.device)
        next_t = (torch.ones(n)*seq[(timestep-1)]).to(var_xt.device)
        at = compute_alpha(diffusion.betas, t.long())
        at_next = compute_alpha(diffusion.betas, next_t.long())
        var_xt_next = (at_next/at) * var_xt

        return var_xt_next
    
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
        fixed_classes = torch.randint(low=args.fixed_class, high=args.fixed_class+1, size=(args.sample_batch_size,)).to(device)

##########  initialize diffusion and model(unet) ########## 
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
    
    fid_dir = f'/data///FID/new_compare/skip/ddim_128/tube_0'
    # fid_dir = f'/data///FID/new_compare/orgin/ddim_128'
    print(f'uq_array is {uq_array}, fid_dir is {fid_dir}')
    var_sum = torch.zeros((args.sample_batch_size, n_rounds)).to(device)
    img_id = 1000000
    samle_batch_size = args.sample_batch_size
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
                
            samle_batch_size = args.sample_batch_size
            xT = fixed_xT[loop*args.sample_batch_size:(loop+1)*args.sample_batch_size, :, :, :].to(device)
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
                    eps_t= sample_from_gaussion(eps_mu_t, eps_var_t)
                    xt_next = singlestep_ddim_sample(diffusion, xt, seq, timestep, eps_t)
                    exp_xt_next = conditioned_exp_iteration(diffusion, exp_xt, seq, timestep, pre_wuq=uq_array[timestep], mc_eps_exp_t=mc_eps_exp_t)
                    var_xt_next = conditioned_var_iteration(diffusion, var_xt, cov_xt_epst, var_epst=eps_var_t, seq=seq, timestep=timestep, pre_wuq= uq_array[timestep])
                    if uq_array[timestep-1] == True:
                        list_xt_next_i, list_eps_mu_t_next_i=[], []
                        for _ in range(mc_sample_size):
                            var_xt_next = torch.clamp(var_xt_next, min=0)
                            xt_next_i = sample_from_gaussion(exp_xt_next, var_xt_next)
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
                    xt_next = singlestep_ddim_sample(diffusion, xt, seq, timestep, eps_mu_t)
                    exp_xt_next = conditioned_exp_iteration(diffusion, exp_xt, seq, timestep, pre_wuq=uq_array[timestep], acc_eps_t = eps_mu_t)
                    var_xt_next = conditioned_var_iteration(diffusion, var_xt, cov_xt_epst= None, var_epst=None, seq= seq, timestep=timestep, pre_wuq= uq_array[timestep])
                    if uq_array[timestep-1] == True:
                        list_xt_next_i, list_eps_mu_t_next_i=[], []
                        for _ in range(mc_sample_size):
                            var_xt_next = torch.clamp(var_xt_next, min=0)
                            xt_next_i = sample_from_gaussion(exp_xt_next, var_xt_next)
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
            
            os.makedirs(os.path.join(fid_dir, 'var/'), exist_ok=True)
            os.makedirs(os.path.join(fid_dir, 'sam/'), exist_ok=True)
            for i in range(x.shape[0]):
                path = os.path.join(fid_dir, 'sam/', f"{img_id}.png")
                var_path = os.path.join(fid_dir, 'var/', f"{img_id}.pt")
                tvu.save_image(x[i], path)
                torch.save(var_xt_next.sum(dim=(1,2,3))[i].cpu(), var_path)
                img_id += 1

        print(f'Sampling {total_n_samples} images in {fid_dir}')
        torch.save(var_sum.cpu(), os.path.join(fid_dir, 'var_sum.pt'))

if __name__ == "__main__":
    main()