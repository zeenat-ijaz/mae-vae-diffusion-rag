import torch

T = 1000

betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


def add_noise(x0, t, noise):
    abar = alpha_bars[t.cpu()].to(x0.device).reshape(-1, 1, 1, 1)
    return abar.sqrt() * x0 + (1 - abar).sqrt() * noise
