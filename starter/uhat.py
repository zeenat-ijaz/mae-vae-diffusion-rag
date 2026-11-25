"""Parts B and C data: UHaT handwritten Urdu characters (28 x 28, grayscale).
Unzip data.zip from Zenodo (or the Kaggle copy) into data/uhat/ first."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _find_split_dir(root, kind, split):
    """Find the folder whose name mentions both `kind` ('char'/'digit') and
    `split` ('train'/'test') and that contains one sub-folder per class."""
    for d in sorted(Path(root).rglob("*")):
        name = d.name.lower()
        if d.is_dir() and kind in name and split in name:
            if any(c.is_dir() for c in d.iterdir()):
                return d
    found = [str(d) for d in Path(root).rglob("*") if d.is_dir()][:20]
    raise FileNotFoundError(f"No '{kind}'/'{split}' folder under {root}. Folders seen: {found}")

def load_uhat_characters(root="data/uhat", split="train", class_names=None):
    """Returns images (N, 1, 28, 28) float32 in [0, 1] with WHITE strokes on a
    BLACK background, labels (N,) int64, and the list of class names.
    Pass the training class_names when loading 'test' so labels match."""
    split_dir = _find_split_dir(root, "char", split)
    if class_names is None:
        class_names = sorted(c.name.lower() for c in split_dir.iterdir() if c.is_dir())
    index = {name: i for i, name in enumerate(class_names)}
    images, labels = [], []
    for cdir in sorted(split_dir.iterdir()):
        if not cdir.is_dir() or cdir.name.lower() not in index:
            continue
        for f in sorted(cdir.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                im = Image.open(f).convert("L").resize((28, 28))
                images.append(np.asarray(im, dtype=np.float32) / 255.0)
                labels.append(index[cdir.name.lower()])
    x = torch.from_numpy(np.stack(images)).unsqueeze(1)
    # Make polarity consistent: if most pixels are bright, the background is
    # white, so invert to get white strokes on black.
    flip = x.mean(dim=(1, 2, 3)) > 0.5
    x[flip] = 1.0 - x[flip]
    return x, torch.tensor(labels, dtype=torch.long), class_names
