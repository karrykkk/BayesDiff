import argparse
import yaml
import os
import torch
import numpy as np

import torch
import warnings

import torchvision.utils as tvu
from torchvision.utils import make_grid
import logging
import time
import copy

def compute_alpha(beta, t):
    # beta is the \beta in ddpm paper
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def singlestep_ddim_sample(model, xt, seq, timestep, eps_t):
    # at.sqrt() is the \alpha_t in our paper
    n = xt.size(0)
    t = (torch.ones(n)*seq[timestep]).to(xt.device)
    next_t = (torch.ones(n)*seq[(timestep-1)]).to(xt.device)
    at = compute_alpha(model.betas, t.long())
    at_next = compute_alpha(model.betas, next_t.long())
    x0_t = (xt - eps_t * (1 - at).sqrt()) / at.sqrt()
    c2 = (1 - at_next).sqrt()
    xt_next = at_next.sqrt() * x0_t + c2 * eps_t

    return xt_next

def var_iteration(model, var_xt, cov_xt_epst, var_epst, seq, timestep):
    # at.sqrt(), st is the \alpha_t, \sigma_t in our paper
    # at is the \alpha_t in ddim paper and is the \bar(\alpha_t) in ddpm paper
    n = var_xt.size(0)
    t = (torch.ones(n)*seq[timestep]).to(var_xt.device)
    next_t = (torch.ones(n)*seq[(timestep-1)]).to(var_xt.device)
    at = compute_alpha(model.betas, t.long())
    at_next = compute_alpha(model.betas, next_t.long())
    st = (1 - at).sqrt()
    st_next = (1 - at_next).sqrt()
    compute_cov_coefficient = (at_next.sqrt()/at.sqrt()) * (st_next - (at_next.sqrt()/at.sqrt()) * st)
    var_xt_next = (at_next/at) * var_xt + 2 * compute_cov_coefficient * cov_xt_epst +\
          torch.square((st_next - (at_next.sqrt()/at.sqrt())*st)) * var_epst
    
    return var_xt_next

def exp_iteration(model, exp_xt, seq, timestep, mc_eps_exp_t):
    n = exp_xt.size(0)
    t = (torch.ones(n)*seq[timestep]).to(exp_xt.device)
    next_t = (torch.ones(n)*seq[(timestep-1)]).to(exp_xt.device)
    at = compute_alpha(model.betas, t.long())
    at_next = compute_alpha(model.betas, next_t.long())
    st = (1 - at).sqrt()
    st_next = (1 - at_next).sqrt()    
    exp_xt_next = (at_next.sqrt()/at.sqrt()) * exp_xt + (st_next - (at_next.sqrt()/at.sqrt()) * st) * mc_eps_exp_t

    return exp_xt_next

def sample_from_gaussion(eps_mu_t, eps_var_t):
    
    # Ls_t = psd_safe_cholesky(eps_var_t)
    samples = eps_mu_t + (torch.randn(eps_mu_t.shape).to(eps_mu_t.device)) * torch.sqrt(eps_var_t)
    
    return samples

