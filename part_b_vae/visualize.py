import sys
sys.path.append(".")
sys.path.append("starter")
sys.path.append("part_b_vae")

import torch
from common import *
from viz import save_grid
from vae_model import VAE

torch.manual_seed(4)
make_folders()

x_train, y_train, x_test, y_test, names = load_uhat()

model = VAE().to(device)
model.load_state_dict(torch.load("weights/vae.pt", map_location=device))
model.eval()

pick = torch.randperm(x_test.shape[0])[:16]
x = x_test[pick].to(device)
with torch.no_grad():
    mu, logvar = model.encode(x)
    recon = torch.sigmoid(model.decode(mu))
grid = torch.cat([x.cpu(), recon.cpu()], dim=0)
save_grid(grid, "results/fig_vae_reconstructions.png", nrow=16)
print("saved results/fig_vae_reconstructions.png (top row: test images, bottom row: reconstructions)")

z = torch.randn(64, 16).to(device)
with torch.no_grad():
    samples = torch.sigmoid(model.decode(z))
save_grid(samples.cpu(), "results/fig_vae_samples.png", nrow=8)
print("saved results/fig_vae_samples.png")

first = 0
second = 1
while y_test[pick[second]] == y_test[pick[first]]:
    second = second + 1
a = x_test[pick[first]].unsqueeze(0).to(device)
b = x_test[pick[second]].unsqueeze(0).to(device)
with torch.no_grad():
    mu_a, _ = model.encode(a)
    mu_b, _ = model.encode(b)
    steps = []
    for i in range(10):
        t = i / 9
        z = (1 - t) * mu_a + t * mu_b
        steps.append(torch.sigmoid(model.decode(z)))
steps = torch.cat(steps, dim=0)
save_grid(steps.cpu(), "results/fig_vae_interpolation.png", nrow=10)
print("interpolation between", names[y_test[pick[first]]], "and", names[y_test[pick[second]]])
print("saved results/fig_vae_interpolation.png")
