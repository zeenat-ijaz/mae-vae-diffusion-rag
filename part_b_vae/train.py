import sys
sys.path.append(".")
sys.path.append("starter")
sys.path.append("part_b_vae")

import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from vae_model import VAE, vae_loss

torch.manual_seed(3)
make_folders()

EPOCHS = 30
BATCH_SIZE = 128

x_train, y_train, x_test, y_test, names = load_uhat()
print("UHaT train:", tuple(x_train.shape), "| test:", tuple(x_test.shape), "| classes:", len(names))

model = VAE().to(device)
print("parameters:", count_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def test_losses():
    model.eval()
    total = 0.0
    recon_sum = 0.0
    kl_sum = 0.0
    with torch.no_grad():
        for idx in batches(x_test.shape[0], 500, shuffle=False):
            x = x_test[idx].to(device)
            logits, mu, logvar = model(x)
            loss, recon, kl = vae_loss(logits, x, mu, logvar)
            n = x.shape[0]
            total += loss.item() * n
            recon_sum += recon.item() * n
            kl_sum += kl.item() * n
    n_all = x_test.shape[0]
    return recon_sum / n_all, kl_sum / n_all, total / n_all


log = {"train_recon": [], "train_kl": [], "train_total": [],
       "test_recon": [], "test_kl": [], "test_total": []}
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    recon_list = []
    kl_list = []
    total_list = []
    for idx in batches(x_train.shape[0], BATCH_SIZE):
        x = x_train[idx].to(device)
        logits, mu, logvar = model(x)
        loss, recon, kl = vae_loss(logits, x, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        recon_list.append(recon.item())
        kl_list.append(kl.item())
        total_list.append(loss.item())

    test_recon, test_kl, test_total = test_losses()
    log["train_recon"].append(mean(recon_list))
    log["train_kl"].append(mean(kl_list))
    log["train_total"].append(mean(total_list))
    log["test_recon"].append(test_recon)
    log["test_kl"].append(test_kl)
    log["test_total"].append(test_total)
    print("epoch", epoch + 1, "| train recon", round(mean(recon_list), 2), "kl", round(mean(kl_list), 2),
          "| test recon", round(test_recon, 2), "kl", round(test_kl, 2), "total", round(test_total, 2))

seconds = time.time() - start_time
print("time in minutes:", round(seconds / 60, 1))

torch.save(model.state_dict(), "weights/vae.pt")

log["params"] = count_params(model)
log["seconds"] = seconds
log["gpu"] = gpu_name()
log["train_images"] = x_train.shape[0]
log["test_images"] = x_test.shape[0]
log["classes"] = len(names)
log["class_names"] = names
save_json(log, "results/vae_train.json")

epochs = range(1, EPOCHS + 1)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(epochs, log["train_recon"], label="train")
plt.plot(epochs, log["test_recon"], label="test")
plt.title("reconstruction term")
plt.xlabel("epoch")
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(epochs, log["train_kl"], label="train")
plt.plot(epochs, log["test_kl"], label="test")
plt.title("KL term")
plt.xlabel("epoch")
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(epochs, log["train_total"], label="train")
plt.plot(epochs, log["test_total"], label="test")
plt.title("negative ELBO")
plt.xlabel("epoch")
plt.legend()
plt.tight_layout()
plt.savefig("results/fig_vae_loss.png", dpi=150)
print("saved results/fig_vae_loss.png")
