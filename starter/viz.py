import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

def save_grid(images, path, nrow=8, titles=None):
    """images: (N, C, H, W) in [0, 1]. Saves an nrow-wide grid to `path`."""
    images = images.detach().cpu().clamp(0, 1)
    n = images.shape[0]
    rows = (n + nrow - 1) // nrow
    fig, axes = plt.subplots(rows, nrow, figsize=(1.4 * nrow, 1.4 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            img = images[i].permute(1, 2, 0).numpy()
            ax.imshow(img.squeeze(-1) if img.shape[-1] == 1 else img, cmap="gray", vmin=0, vmax=1)
            if titles is not None:
                ax.set_title(titles[i], fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
