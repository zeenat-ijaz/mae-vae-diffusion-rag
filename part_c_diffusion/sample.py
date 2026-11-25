import sys
sys.path.append(".")
sys.path.append("starter")
sys.path.append("part_c_diffusion")

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from viz import save_grid
from schedule import T, betas, alphas, alpha_bars, add_noise
from unet import make_unet


def sample_images(unet, labels, keep_steps=None):
    n = labels.shape[0]
    x = torch.randn(n, 1, 28, 28, device=device)
    saved = {}
    with torch.no_grad():
        for t in reversed(range(T)):
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)
            predicted = unet(x, t_batch, class_labels=labels).sample
            alpha = alphas[t].item()
            abar = alpha_bars[t].item()
            beta = betas[t].item()
            x = (1 / alpha ** 0.5) * (x - (1 - alpha) / (1 - abar) ** 0.5 * predicted)
            if t > 0:
                x = x + (beta ** 0.5) * torch.randn_like(x)
            if keep_steps is not None and t in keep_steps:
                saved[t] = x.clamp(-1, 1).cpu()
    return x.clamp(-1, 1), saved


def to_01(x):
    return (x + 1) / 2


if __name__ == "__main__":
    torch.manual_seed(6)
    make_folders()

    x_train, y_train, x_test, y_test, names = load_uhat()

    unet = make_unet().to(device)
    unet.load_state_dict(torch.load("weights/unet.pt", map_location=device))
    unet.eval()

    image = x_train[0:1] * 2 - 1
    times = [0, 100, 250, 500, 750, 999]
    noised = []
    noise = torch.randn_like(image)
    for t in times:
        noised.append(add_noise(image, torch.tensor([t]), noise))
    noised = torch.cat(noised, dim=0).clamp(-1, 1)
    save_grid(to_01(noised), "results/fig_diffusion_forward.png", nrow=6,
              titles=["t=" + str(t) for t in times])
    print("saved results/fig_diffusion_forward.png")

    labels = torch.arange(40).repeat_interleave(8).to(device)
    all_samples = []
    for start in range(0, 320, 160):
        samples, saved = sample_images(unet, labels[start:start + 160])
        all_samples.append(samples.cpu())
        print("sampled", start + 160, "of 320")
    all_samples = to_01(torch.cat(all_samples, dim=0))

    fig, axes = plt.subplots(40, 8, figsize=(8, 40))
    for row in range(40):
        for col in range(8):
            ax = axes[row, col]
            ax.imshow(all_samples[row * 8 + col, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(names[row], rotation=0, ha="right", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("results/fig_diffusion_samples.png", dpi=100)
    plt.close()
    print("saved results/fig_diffusion_samples.png")

    keep = [999, 750, 500, 250, 100, 0]
    one_label = torch.tensor([0], device=device)
    final, saved = sample_images(unet, one_label, keep_steps=keep)
    steps = torch.cat([saved[t] for t in keep], dim=0)
    save_grid(to_01(steps), "results/fig_diffusion_denoising.png", nrow=6,
              titles=["t=" + str(t) for t in keep])
    print("saved results/fig_diffusion_denoising.png (character: " + names[0] + ")")
