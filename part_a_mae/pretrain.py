import sys
sys.path.append(".")
sys.path.append("starter")
sys.path.append("part_a_mae")

import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from eurosat import load_eurosat_split, patchify
from mae_model import MAE

torch.manual_seed(0)
make_folders()

EPOCHS = 100
BATCH_SIZE = 256

images, labels = load_eurosat_split("train")
patches = patchify(images)
print("training images:", patches.shape[0], "| patches per image:", patches.shape[1])

model = MAE().to(device)
print("parameters:", count_params(model))

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)

epoch_losses = []
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    losses = []
    for idx in batches(patches.shape[0], BATCH_SIZE):
        batch = patches[idx].to(device)
        loss, pred, mask = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    epoch_loss = mean(losses)
    epoch_losses.append(epoch_loss)
    print("epoch", epoch + 1, "| loss", round(epoch_loss, 5))

seconds = time.time() - start_time
print("time in minutes:", round(seconds / 60, 1))

torch.save(model.state_dict(), "weights/mae.pt")

save_json({"epoch_losses": epoch_losses, "params": count_params(model), "seconds": seconds,
           "gpu": gpu_name(), "train_images": patches.shape[0]}, "results/mae_pretrain.json")

plt.figure(figsize=(6, 4))
plt.plot(range(1, EPOCHS + 1), epoch_losses)
plt.xlabel("epoch")
plt.ylabel("masked MSE loss")
plt.title("MAE pretraining loss")
plt.tight_layout()
plt.savefig("results/fig_mae_loss.png", dpi=150)
print("saved results/fig_mae_loss.png")
