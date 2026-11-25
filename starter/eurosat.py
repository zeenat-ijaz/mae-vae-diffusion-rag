"""Part A data: EuroSAT RGB satellite tiles (64 x 64, 10 land-cover classes)."""
import numpy as np
import torch
from datasets import load_dataset

EUROSAT_REPO = "blanchon/EuroSAT_RGB"
CLASS_NAMES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
               "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
PATCH = 8                      # 64 / 8 = 8 x 8 = 64 patches per image

def load_eurosat_split(split):
    """split in {'train', 'validation', 'test'}.
    Returns images (N, 3, 64, 64) float32 in [0, 1] and labels (N,) int64."""
    ds = load_dataset(EUROSAT_REPO, split=split)
    imgs = np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in ds["image"]])
    images = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
    labels = torch.tensor(ds["label"], dtype=torch.long)
    return images, labels

def few_label_indices(labels, per_class=16, seed=42):
    """Indices of a fixed labelled subset: `per_class` images of each class.
    Every group gets the same 160 images."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(len(CLASS_NAMES)):
        members = np.flatnonzero(labels.numpy() == c)
        idx.extend(rng.choice(members, size=per_class, replace=False).tolist())
    return torch.tensor(sorted(idx))

def patchify(images, p=PATCH):
    """(N, C, H, W) -> (N, L, p*p*C) with L = (H/p)*(W/p), patches in row-major order."""
    N, C, H, W = images.shape
    x = images.reshape(N, C, H // p, p, W // p, p)
    x = x.permute(0, 2, 4, 3, 5, 1)                 # N, H/p, W/p, p, p, C
    return x.reshape(N, (H // p) * (W // p), p * p * C)

def unpatchify(patches, p=PATCH, C=3):
    """Inverse of patchify: (N, L, p*p*C) -> (N, C, H, W)."""
    N, L, _ = patches.shape
    h = w = int(L ** 0.5)
    x = patches.reshape(N, h, w, p, p, C).permute(0, 5, 1, 3, 2, 4)
    return x.reshape(N, C, h * p, w * p)
