import sys
sys.path.append(".")
sys.path.append("starter")
sys.path.append("part_c_diffusion")

import time
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from schedule import T, add_noise
from unet import make_unet

torch.manual_seed(5)
make_folders()

EPOCHS = 40
BATCH_SIZE = 128

x_train, y_train, x_test, y_test, names = load_uhat()
x_train = x_train * 2 - 1
print("UHaT train:", tuple(x_train.shape), "| classes:", len(names))

unet = make_unet().to(device)
print("parameters:", count_params(unet))
optimizer = torch.optim.Adam(unet.parameters(), lr=2e-4)

epoch_losses = []
start_time = time.time()

for epoch in range(EPOCHS):
    unet.train()
    losses = []
    for idx in batches(x_train.shape[0], BATCH_SIZE):
        x0 = x_train[idx].to(device)
        y = y_train[idx].to(device)
        t = torch.randint(0, T, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        x_t = add_noise(x0, t, noise)
        predicted = unet(x_t, t, class_labels=y).sample
        loss = F.mse_loss(predicted, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    epoch_loss = mean(losses)
    epoch_losses.append(epoch_loss)
    print("epoch", epoch + 1, "| loss", round(epoch_loss, 5))
    torch.save(unet.state_dict(), "weights/unet.pt")

seconds = time.time() - start_time
print("time in minutes:", round(seconds / 60, 1))

save_json({"epoch_losses": epoch_losses, "params": count_params(unet), "seconds": seconds,
           "gpu": gpu_name(), "train_images": x_train.shape[0], "test_images": x_test.shape[0],
           "classes": len(names), "class_names": names}, "results/diffusion_train.json")

plt.figure(figsize=(6, 4))
plt.plot(range(1, EPOCHS + 1), epoch_losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Diffusion training loss")
plt.tight_layout()
plt.savefig("results/fig_diffusion_loss.png", dpi=150)
print("saved results/fig_diffusion_loss.png")
