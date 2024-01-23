import argparse
import yaml
import os
import torch
import numpy as np

from runners.diffusion import Diffusion
from models.diffusion import Model, flattened_Model
from models.improved_ddpm.unet import UNetModel as ImprovedDDPM_Model
from models.guided_diffusion.unet import UNetModel as GuidedDiffusion_Model
from models.ema import EMAHelper
from custom_model import CustomModel
import torch
import warnings

from la_train_datasets import celeba_dataset, imagenet_dataset
import torchvision.utils as tvu
from torchvision.utils import make_grid
import logging
import time
import copy

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

def singlestep_ddim_sample(diffusion, xt, seq, timestep, eps_t):
    # at.sqrt() is the \alpha_t in our paper
    n = xt.size(0)
    t = (torch.ones(n)*seq[timestep]).to(xt.device)
    next_t = (torch.ones(n)*seq[(timestep-1)]).to(xt.device)
    at = compute_alpha(diffusion.betas, t.long())
    at_next = compute_alpha(diffusion.betas, next_t.long())
    x0_t = (xt - eps_t * (1 - at).sqrt()) / at.sqrt()
    c2 = (1 - at_next).sqrt()
    xt_next = at_next.sqrt() * x0_t + c2 * eps_t

    return xt_next

def var_iteration(diffusion, var_xt, cov_xt_epst, var_epst, seq, timestep):
    # at.sqrt(), st is the \alpha_t, \sigma_t in our paper
    # at is the \alpha_t in ddim paper and is the \bar(\alpha_t) in ddpm paper
    n = var_xt.size(0)
    t = (torch.ones(n)*seq[timestep]).to(var_xt.device)
    next_t = (torch.ones(n)*seq[(timestep-1)]).to(var_xt.device)
    at = compute_alpha(diffusion.betas, t.long())
    at_next = compute_alpha(diffusion.betas, next_t.long())
    st = (1 - at).sqrt()
    st_next = (1 - at_next).sqrt()
    compute_cov_coefficient = (at_next.sqrt()/at.sqrt()) * (st_next - (at_next.sqrt()/at.sqrt()) * st)
    var_xt_next = (at_next/at) * var_xt + 2 * compute_cov_coefficient * cov_xt_epst +\
          torch.square((st_next - (at_next.sqrt()/at.sqrt())*st)) * var_epst
    
    return var_xt_next

def exp_iteration(diffusion, exp_xt, seq, timestep, mc_eps_exp_t):
    n = exp_xt.size(0)
    t = (torch.ones(n)*seq[timestep]).to(exp_xt.device)
    next_t = (torch.ones(n)*seq[(timestep-1)]).to(exp_xt.device)
    at = compute_alpha(diffusion.betas, t.long())
    at_next = compute_alpha(diffusion.betas, next_t.long())
    st = (1 - at).sqrt()
    st_next = (1 - at_next).sqrt()    
    exp_xt_next = (at_next.sqrt()/at.sqrt()) * exp_xt + (st_next - (at_next.sqrt()/at.sqrt()) * st) * mc_eps_exp_t

    return exp_xt_next

def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        L = torch.linalg.cholesky(A, upper=upper, out=out)
        return L
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            raise NanError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(10):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.linalg.cholesky(Aprime, upper=upper, out=out)
                warnings.warn(
                    f"A not p.d., added jitter of {jitter_new} to the diagonal",
                    RuntimeWarning,
                )
                return L
            except RuntimeError:
                continue
        # return torch.randn_like(A).tril()
        raise e

def sample_from_gaussion(eps_mu_t, eps_var_t):
    
    # Ls_t = psd_safe_cholesky(eps_var_t)
    samples = eps_mu_t + (torch.randn(eps_mu_t.shape).to(eps_mu_t.device)) * torch.sqrt(eps_var_t)
    
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
