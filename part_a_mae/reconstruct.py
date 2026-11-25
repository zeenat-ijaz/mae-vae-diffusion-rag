import sys
sys.path.append(".")
sys.path.append("starter")
sys.path.append("part_a_mae")

import torch
from common import *
from viz import save_grid
from eurosat import load_eurosat_split, patchify, unpatchify
from mae_model import MAE

torch.manual_seed(1)
make_folders()

images, labels = load_eurosat_split("test")
model = MAE().to(device)
model.load_state_dict(torch.load("weights/mae.pt", map_location=device))
model.eval()

original = images[:8]
patches = patchify(original)

with torch.no_grad():
    loss, pred, mask = model(patches.to(device))
pred = pred.cpu()
mask = mask.cpu()

mask3 = mask.unsqueeze(-1)
masked_input = patches * (1 - mask3) + 0.5 * mask3
reconstruction = patches * (1 - mask3) + pred * mask3

row1 = unpatchify(masked_input)
row2 = unpatchify(reconstruction)
row3 = original

grid = torch.cat([row1, row2, row3], dim=0)
save_grid(grid, "results/fig_mae_reconstructions.png", nrow=8)
print("saved results/fig_mae_reconstructions.png (rows: masked input, reconstruction, original)")
