import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def make_beta_schedule(schedule, n_timestep, betas_1=1e-4, beats_T=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = torch.linspace(betas_1 ** 0.5, beats_T ** 0.5,
                            n_timestep, dtype=torch.float64) ** 2
    elif schedule == 'linear':
        betas = torch.linspace(betas_1, beats_T,
                            n_timestep, dtype=torch.float64)
    elif schedule == 'const':
        betas = beats_T * torch.ones(n_timestep, dtype=torch.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / torch.linspace(n_timestep,
                                 1, n_timestep, dtype=torch.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, schedule, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.betas = make_beta_schedule(schedule=schedule, n_timestep=T, betas_1=beta_1, beats_T=beta_T)
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, pan_x_0):
        """
        Algorithm 1.
        pan : 64x64
        pan : 64x64
        """
        t = torch.randint(self.T, size=(pan_x_0.shape[0], ), device=pan_x_0.device)
        noise = torch.randn_like(pan_x_0)
        pan_x_t = (
            extract(self.sqrt_alphas_bar, t, pan_x_0.shape) * pan_x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, pan_x_0.shape) * noise)
        loss = F.mse_loss(self.model(pan_x_t, t), noise, reduction='none')
        return loss
