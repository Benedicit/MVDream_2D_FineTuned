import math
import torch
import einops
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
import torch.nn.functional as F
from typing import Union
from functools import partial
from mvdream.ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor
from torchdiffeq import odeint
from mvdream.ldm.interface import LatentDiffusionInterface
'''
Based on the Code of Diff2Flow https://github.com/CompVis/diff2flow/blob/main/diff2flow

'''

def forward_with_cfg(x, t, model : LatentDiffusionInterface, cond, cfg_scale=1.0, uc_cond=None):
    '''
    Adapted from DDIM sampler of MVDream and Diff2Flow
    '''
    if cfg_scale == 1.0:
        model_output = model.apply_model(x, t, cond)
    else:
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        if isinstance(cond, dict):
            assert isinstance(uc_cond, dict)
            c_in = dict()
            for k in cond:
                if isinstance(cond[k], list):
                    c_in[k] = [torch.cat([
                        uc_cond[k][i],
                        cond[k][i]]) for i in range(len(cond[k]))]
                elif isinstance(cond[k], torch.Tensor):
                    c_in[k] = torch.cat([
                        uc_cond[k],
                        cond[k]])
                else:
                    assert cond[k] == uc_cond[k]
                    c_in[k] = cond[k]
        elif isinstance(cond, list):
            c_in = list()
            assert isinstance(uc_cond, list)
            for i in range(len(cond)):
                c_in.append(torch.cat([uc_cond[i], cond[i]]))
        else:
            c_in = torch.cat([uc_cond, cond])
        model_uncond, model_t = model.apply_model(x_in, t_in, c_in).chunk(2)
        model_output = model_uncond + cfg_scale * (model_t - model_uncond)
    return model_output

# default from https://github.com/willisma/SiT
_ATOL = 1e-6
_RTOL = 1e-3

def pad_v_like_x(v_, x_) -> Tensor | float:
    """
    Function to reshape the vector by the number of dimensions
    of x. E.g. x (bs, c, h, w), v (bs) -> v (bs, 1, 1, 1).
    """
    if isinstance(v_, float):
        return v_
    return v_.reshape(-1, *([1] * (x_.ndim - 1)))

def enforce_zero_terminal_snr(betas):
    # Copied from https://openaccess.thecvf.com/content/WACV2024/papers/Lin_Common_Diffusion_Noise_Schedules_and_Sample_Steps_Are_Flawed_WACV_2024_paper.pdf
    # Convert betas to alphas_bar_sqrt
    if isinstance(betas, np.ndarray):
        betas = torch.tensor(betas)
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas.numpy()

def extract_and_interpolate_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    # t can be float here, linearly interpolate between left and right index
    t = t.clamp(0, a.shape[-1] - 1)
    left_idx = t.long()
    right_idx = (left_idx + 1).clamp(max=a.shape[-1] - 1)
    left_val = a.gather(-1, left_idx)
    right_val = a.gather(-1, right_idx)
    t_ = t - left_idx.float()
    out = left_val * (1 - t_) + right_val * t_
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class LinearSchedule:
    def alpha_t(self, t):
        return t

    def alpha_dt_t(self, t):
        return 1

    def sigma_t(self, t):
        return 1 - t

    def sigma_dt_t(self, t):
        return -1

    """ Legacy functions to work with SiT Sampler """

    def compute_alpha_t(self, t):
        return self.alpha_t(t), self.alpha_dt_t(t)

    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return self.sigma_t(t), self.sigma_dt_t(t)

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        return 1 / t

    def compute_drift(self, x, t):
        """We always output sde according to score parametrization; """
        t = pad_v_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t

        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        """Compute the diffusion term of the SDE
        Args:
          x: [batch_dim, ...], data point
          t: [batch_dim,], time vector
          form: str, form of the diffusion term
          norm: float, norm of the diffusion term
        """
        t = pad_v_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * torch.cos(np.pi * t) + 1) ** 2,
            "increasing-decreasing": norm * torch.sin(np.pi * t) ** 2,
        }

        try: diffusion = choices[form]
        except KeyError: raise NotImplementedError(f"Diffusion form {form} not implemented")

        return diffusion

    def get_score_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to score
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = pad_v_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score

    def get_noise_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to denoiser
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = pad_v_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = reverse_alpha_ratio * d_sigma_t - sigma_t
        noise = (reverse_alpha_ratio * velocity - mean) / var
        return noise

    def get_velocity_from_score(self, score, x, t):
        """Wrapper function: transfrom score prediction model to velocity
        Args:
            score: [batch_dim, ...] shaped tensor; score model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = pad_v_like_x(t, x)
        drift, var = self.compute_drift(x, t)
        velocity = var * score - drift
        return velocity

class GVPSchedule(LinearSchedule):
    def alpha_t(self, t):
        return torch.sin(t * math.pi / 2)

    def alpha_dt_t(self, t):
        return 0.5 * math.pi * torch.cos(t * math.pi / 2)

    def sigma_t(self, t):
        return torch.cos(t * math.pi / 2)

    def sigma_dt_t(self, t):
        return - 0.5 * math.pi * torch.sin(t * math.pi / 2)

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return np.pi / (2 * torch.tan(t * np.pi / 2))






class FlowMatching(nn.Module):
    def __init__(self, device,model,diffusion_parameterization: str = 'eps', diffusion_schedule: str = 'linear',):
        super().__init__()
        self.device = device
        self.sigma_min = 0.0
        self.model = model
        self.schedule = LinearSchedule()
        self.diffusion_parameterization = diffusion_parameterization
        self.register_sdv2_schedule(diffusion_schedule)

    def register_sdv2_schedule(self, diffusion_schedule, enforce_zero_snr=False):
        linear_start = 0.00085
        linear_end = 0.0120

        betas = make_beta_schedule(
            diffusion_schedule,
            n_timestep=1000,
            linear_start=linear_start,
            linear_end=linear_end,
        )
        if enforce_zero_snr:
            betas = enforce_zero_terminal_snr(betas)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        alphas_cumprod_full = np.append(1., alphas_cumprod)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        # self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('alphas_cumprod_full', to_torch(alphas_cumprod_full))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('sqrt_alphas_cumprod_full', to_torch(np.sqrt(alphas_cumprod_full)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod_full', to_torch(np.sqrt(1. - alphas_cumprod_full)))

        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        self.register_buffer('rectified_alphas_cumprod_full', self.sqrt_alphas_cumprod_full / (self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full))
        self.register_buffer('rectified_sqrt_alphas_cumprod_full', self.sqrt_one_minus_alphas_cumprod_full / (self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full))

    def compute_xt(self, x0: Tensor, x1: Tensor, t: Tensor):
        """
        Sample from the time-dependent density p_t
            xt ~ N(alpha_t * x1 + sigma_t * x0, sigma_min * I),
        according to Eq. (1) in [3] and for the linear schedule Eq. (14) in [2].

        Args:
            x0 : shape (bs, *dim), represents the source minibatch (noise)
            x1 : shape (bs, *dim), represents the target minibatch (data)
            t  : shape (bs,) represents the time in [0, 1]
        Returns:
            xt : shape (bs, *dim), sampled point along the time-dependent density p_t
        """
        t = pad_v_like_x(t, x0)
        alpha_t = self.schedule.alpha_t(t)
        sigma_t = self.schedule.sigma_t(t)
        xt = alpha_t * x1 + sigma_t * x0
        if self.sigma_min > 0:
            xt += self.sigma_min * torch.randn_like(xt)
        return xt

    def compute_ut(self, x0: Tensor, x1: Tensor, t: Tensor):
        """
        Compute the time-dependent conditional vector field
            ut = alpha_dt_t * x1 + sigma_dt_t * x0,
        see Eq. (7) in [3].

        Args:
            x0 : Tensor, shape (bs, *dim), represents the source minibatch (noise)
            x1 : Tensor, shape (bs, *dim), represents the target minibatch (data)
            t  : FloatTensor, shape (bs,) represents the time in [0, 1]
        Returns:
            ut : conditional vector field
        """
        t = pad_v_like_x(t, x0)
        alpha_dt_t = self.schedule.alpha_dt_t(t)
        sigma_dt_t = self.schedule.sigma_dt_t(t)
        return alpha_dt_t * x1 + sigma_dt_t * x0

    def sample_vt(self, fm_x, fm_t, cond, uc_cond=None, cfg_scale=1.0,):
        """
        Sample the v-parameterized vector field at time t
        """
        dm_t = self.convert_fm_t_to_dm_t(fm_t)
        #print("ALAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARM")
        #print(fm_t, dm_t)
        dm_t_int = dm_t.long()
        dm_x = self.convert_fm_xt_to_dm_xt(fm_x, fm_t)
        # vt = self.net(dm_x, dm_t, **kwargs)

        vt = forward_with_cfg(dm_x, dm_t, self.model, cond=cond, uc_cond=uc_cond, cfg_scale=cfg_scale,)
        #vt = self.model.apply_model(dm_x, dm_t_int, cond)

        # Maybe there is a better way to handle NAN values
        if torch.isnan(vt).any():
            vt[torch.isnan(vt)] = 0

        # vt = self.forward(x=dm_x, t=dm_t, **kwargs)
        if self.diffusion_parameterization == 'v':
            vector_field = self.get_vector_field_from_v(vt, dm_x, dm_t_int)
        elif self.diffusion_parameterization == 'eps':
            vector_field = self.get_vector_field_from_eps(vt, dm_x, dm_t_int)
        else:
            raise NotImplementedError('unknown diffusion parameterization')
        return vector_field

    def convert_fm_t_to_dm_t(self, t):
        """
        Convert the continuous time t in [0,1] to discrete time t [0, 1000)
        # TODO: Make it compatible with zero-terminal SNR
        """
        rectified_alphas_cumprod_full = self.rectified_alphas_cumprod_full.clone().to(t.device)
        # reverse the rectified_alphas_cumprod_full for searchsorted
        rectified_alphas_cumprod_full = torch.flip(rectified_alphas_cumprod_full, [0])

        right_index = torch.searchsorted(rectified_alphas_cumprod_full, t, right=True)
        left_index = right_index - 1
        right_value = rectified_alphas_cumprod_full[right_index]
        left_value = rectified_alphas_cumprod_full[left_index]

        denom = right_value - left_value
        denom = torch.where(denom == 0, torch.ones_like(denom) * 1e-6, denom)

        #dm_t = left_index + (t - left_value) / (right_value - left_value)
        dm_t = left_index + (t - left_value) / denom
        # now reverse back the dm_t

        dm_t = self.num_timesteps - dm_t
        dm_t = dm_t.clamp(0, self.num_timesteps - 1)
        return dm_t

    def convert_fm_xt_to_dm_xt(self, fm_xt, fm_t):
        """
        Convert fm trajectory to dm trajectory using the fm t
        We use linear scaling here
        """
        scale = self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full
        dm_t = self.convert_fm_t_to_dm_t(fm_t)
        # do lienar interpolation here
        dm_t_left_index = torch.floor(dm_t)
        dm_t_right_index = torch.ceil(dm_t)
        dm_t_left_value = scale[dm_t_left_index.long()]
        dm_t_right_value = scale[dm_t_right_index.long()]

        scale_t = dm_t_left_value + (dm_t - dm_t_left_index) * (dm_t_right_value - dm_t_left_value)
        scale_t = scale_t.view(-1, 1, 1, 1)
        dm_xt = fm_xt * scale_t
        return dm_xt

    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def predict_start_from_eps(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def get_vector_field_from_v(self, v, x_t, t):
        """
        v is the SD v-parameterized vector field with v = sqrt(alpha_cumprod) * eps - sqrt(1 - alpha_cumprod) * z
        the FM vector field is defined as z - eps

        First of all convert the x_t from the rectified flow trajectory to the original diffusion trajectory
        Then calculate the vector field from the v-parameterized vector field
        """
        z_pred = self.predict_start_from_z_and_v(x_t, t, v)
        eps_pred = self.predict_eps_from_z_and_v(x_t, t, v)
        vector_field = z_pred - eps_pred                    # z - eps
        return vector_field

    def get_vector_field_from_eps(self, noise, x_t, t):
        """
        eps is the SD eps-parameterized vector field with
        the FM vector field is defined as z - eps

        First of all convert the x_t from the rectified flow trajectory to the original diffusion trajectory
        Then calculate the vector field from the eps-parameterized vector field
        """
        z_pred = self.predict_start_from_eps(x_t, t, noise)
        eps_pred = noise
        vector_field = z_pred - eps_pred                    # z - eps
        return vector_field

    def forward(self, x, t, cond, uc_cond=None, cfg_scale=1.0,):
        """
        Forward pass for the flow model
        """
        if t.numel() == 1:
            t = t.expand(x.shape[0])
        _pred = self.sample_vt(x, t, cond, uc_cond=uc_cond, cfg_scale=cfg_scale,)
        return _pred

    def ode_fn(self, t, x, cond, uc_cond, cfg_scale):
        if t.numel() == 1:
            t = t.expand(x.shape[0])
        _pred = self.sample_vt(x, t,cond=cond, uc_cond=uc_cond, cfg_scale=cfg_scale,)
        return _pred


    def generate(self, x: Tensor, sample_kwargs=None, reverse=False, return_intermediates=False, **kwargs):
        """
        Args:
            x: source minibatch (bs, *dim)
            sample_kwargs: dict, additional sampling arguments for the solver
                num_steps: int, number of steps to take
                cfg_scale: float, scale for the classifier-free guidance
                cond: torch.Tensor, conditioning information like cameras etc.
                uc_cond: torch.Tensor, unconditional conditioning information (1, *dim) or (bs, *dim)
                intermediate_freq: int, frequency of intermediate outputs
                use_sde: if true, use SDE sampling instead of ODE
                __ ODE Sampler __:
                    method: str, method for the ODE solver (see torchdiffeq)
                    atol/rtol: float, absolute and relative tolerance for the ODE solver
                __ SDE Sampler __:
                    method: str, method for the SDE solver (euler, heun)
                    diffusion_form: str, form of the diffusion coefficient (sigma, SBDM, ...)
                    diffusion_norm: float, magnitude of the diffusion coefficient (default 1.0)
                    last_step: str, type of the last step (Mean, Tweedie, Euler)
                    last_step_size: float, size of the last step (default 0.04)
                    progress: bool, whether to show a progress bar
            reverse: bool, whether to reverse the direction of the flow. If True,
                we map from x1 -> x0, otherwise we map from x0 -> x1.
            n_intermediates: int, number of intermediate points to return.
            kwargs: additional arguments for the network
        """
        sample_kwargs = sample_kwargs or {}

        # timesteps
        num_steps = sample_kwargs.get("num_steps", 50)
        t = torch.linspace(0, 1, num_steps, dtype=x.dtype).to(x.device)
        t = 1 - t if reverse else t

        # include classifier-free guidance
        cfg_scale = sample_kwargs.get("cfg_scale", 1.0)
        uc_cond = sample_kwargs.get("uc_cond", None)
        cond = sample_kwargs.get("cond")

        if sample_kwargs.get("use_sde", False):
            results = self.sde_sampler.sample(
                init=x,
                model=self.model,                         # sde_sampler already includes CFG
                sampling_method=sample_kwargs.get("method", "euler"),
                diffusion_form=sample_kwargs.get("diffusion_form", "sigma"),
                diffusion_norm=sample_kwargs.get("diffusion_norm", 1.0),
                last_step=sample_kwargs.get("last_step", "Mean"),
                last_step_size=sample_kwargs.get("last_step_size", 0.04),
                num_steps=num_steps,
                progress=sample_kwargs.get("progress", False),
                return_intermediates=True,
                cond=cond,
                uc_cond=uc_cond,
                cfg_scale=cfg_scale,
                **kwargs
            )

        # ODE sampling
        else:
            method = sample_kwargs.get("method", "euler")

            ode_fn = partial(self.ode_fn,cond=cond, uc_cond=uc_cond, cfg_scale=cfg_scale,)

            if method == "euler":
                delta_t = 1 / num_steps
                pred = x.clone().to(self.device)
                intermediates = [pred]

                for i in tqdm(range(num_steps), disable=not sample_kwargs.get("progress", True), desc="ODE sampling"):
                    t = torch.ones(x.shape[0], device=x.device) * delta_t * i
                    v = ode_fn(t=t, x=pred)
                    pred = pred + delta_t * v

                    if return_intermediates:
                        intermediates.append(pred.cpu())

                results = [pred] if not return_intermediates else intermediates

            else:
                t = torch.linspace(0, 1, num_steps + 1, dtype=x.dtype).to(x.device)
                t = 1 - t if reverse else t
                results = odeint(
                    ode_fn,
                    x,
                    t,
                    method=method,
                    atol=sample_kwargs.get("atol", _ATOL),
                    rtol=sample_kwargs.get("rtol", _RTOL)
                )

        if return_intermediates:
            intermediate_freq = sample_kwargs.get("intermediate_freq", 5)
            results = torch.stack([results[0], *results[1:-1:intermediate_freq], results[-1]], 0)
            return results
        return results[-1]

    def training_losses(self, x1: Tensor, cond, x0: Tensor = None, t = None) -> Tensor:
        """
        Args:
            t: time step
            x1: shape (bs, *dim), represents the target minibatch (data)
            x0: shape (bs, *dim), represents the source minibatch, if None
                we sample x0 from a standard normal distribution.
            cond: additional arguments for the conditional flow
                network (e.g. conditioning information)
        Returns:
            loss: scalar, the training loss for the flow model
        """
        if x0 is None:
            x0 = torch.randn_like(x1)

        bs, dev, dtype = x1.shape[0], x1.device, x1.dtype

        # Sample time t from uniform distribution U(0, 1)
        if t is None:
            t = torch.rand(bs, device=dev, dtype=dtype)

        # sample xt and ut
        xt = self.compute_xt(x0=x0, x1=x1, t=t)
        ut = self.compute_ut(x0=x0, x1=x1, t=t)
        vt = self.sample_vt(fm_x=xt, fm_t=t, cond=cond)

        #return F.mse_loss(vt, ut)
        return (vt - ut).square().mean()
