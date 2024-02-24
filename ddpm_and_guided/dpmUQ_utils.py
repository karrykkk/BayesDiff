import torch

def get_model_input_time(ns, t_continuous):
    """
    Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
    For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
    For continuous-time DPMs, we just use `t_continuous`.
    """
    if ns.schedule == 'discrete':
        return (t_continuous - 1. / ns.total_N) * 1000.
    else:
        return t_continuous
    
def sample_from_gaussion(eps_mu_t, eps_var_t):
    
    # Ls_t = psd_safe_cholesky(eps_var_t)
    samples = eps_mu_t + (torch.randn(eps_mu_t.shape).to(eps_mu_t.device)) * torch.sqrt(eps_var_t)
    
    return samples

def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)

def singlestep_dpm_solver_second_update(ns, x, s, t, custom_uvit, model_s, r1=0.5, **model_kwargs):
    """
    Singlestep solver DPM-Solver-2 from time `s` to time `t`.

    Args:
        x: A pytorch tensor. The initial value at time `s`.
        s: A pytorch tensor. The starting time, with the shape (1,).
        t: A pytorch tensor. The ending time, with the shape (1,).
        r1: A `float`. The hyperparameter of the second-order solver.
        model_s: A pytorch tensor. The model function evaluated at time `s`.
            If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
        return_intermediate: A `bool`. If true, also return the model value at time `s` and `s1` (the intermediate time).
        solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
            The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
    Returns:
        x_t: A pytorch tensor. The approximated solution at time `t`.
    """

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
    model_s1, model_s1_var = custom_uvit(x_s1, input_s1.expand(x_s1.shape[0]) , **model_kwargs)

    sample_model_s1 = sample_from_gaussion(model_s1, model_s1_var)

    x_t = (
        torch.exp(log_alpha_t - log_alpha_s) * x
        - (sigma_t * phi_1) * model_s
        - (0.5 / r1) * (sigma_t * phi_1) * (sample_model_s1 - model_s)
    )

    return x_t, model_s1, model_s1_var

def exp_iteration(exp_xt, ns, s, t, mc_eps_exp_s1, r1=0.5):
    lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
    h = lambda_t - lambda_s
    sigma_t = ns.marginal_std(t)
    log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
    phi_1 = torch.expm1(h)
    exp_xt_next = torch.exp(log_alpha_t - log_alpha_s) * exp_xt - (sigma_t * phi_1) * mc_eps_exp_s1

    return exp_xt_next

def origin_singlestep_dpm_solver_second_update(ns, x, s, t, custom_uvit, model_s, r1=0.5, **model_kwargs):
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

    return x_t

def var_iteration(var_xt, ns, s, t, cov_xt_epst, var_epst, r1=0.5):
    lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
    h = lambda_t - lambda_s
    sigma_t = ns.marginal_std(t)
    log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
    phi_1 = torch.expm1(h)
    var_xt_next = torch.square(torch.exp(log_alpha_t - log_alpha_s)) * var_xt - \
        2*torch.exp(log_alpha_t - log_alpha_s)*(sigma_t * phi_1)*cov_xt_epst + torch.square(sigma_t * phi_1) * var_epst
    
    return var_xt_next

class NoiseScheduleVP:
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
        ):
        """Create a wrapper class for the forward SDE (VP type).

        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:

            t = self.inverse_lambda(lambda_t)

        ===============================================================

        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).

        1. For discrete-time DPMs:

            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.

            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)

            Note that we always have alphas_cumprod = cumprod(1 - betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.

            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \sqrt{\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).


        2. For continuous-time DPMs:

            We support the linear VPSDE for the continuous time setting. The hyperparameters for the noise
            schedule are the default settings in Yang Song's ScoreSDE:

            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                T: A `float` number. The ending time of the forward process.

        ===============================================================

        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).
        
        ===============================================================

        Example:

        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', betas=betas)

        # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)

        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)

        """

        if schedule not in ['discrete', 'linear']:
            raise ValueError("Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear'".format(schedule))

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.T = 1.
            self.log_alpha_array = self.numerical_clip_alpha(log_alphas).reshape((1, -1,)).to(dtype=dtype)
            self.total_N = self.log_alpha_array.shape[1]
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
        else:
            self.T = 1.
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1

    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        """
        For some beta schedules such as cosine schedule, the log-SNR has numerical isssues. 
        We clip the log-SNR near t=T within -5.1 to ensure the stability.
        Such a trick is very useful for diffusion models with the cosine schedule, such as i-DDPM, guided-diffusion and GLIDE.
        """
        log_sigmas = 0.5 * torch.log(1. - torch.exp(2. * log_alphas))
        lambs = log_alphas - log_sigmas  
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))
        
def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand
