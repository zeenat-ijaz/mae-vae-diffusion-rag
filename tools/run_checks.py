import sys
sys.path.append(".")
sys.path.append("starter")
sys.path.append("part_a_mae")
sys.path.append("part_b_vae")
sys.path.append("part_c_diffusion")
sys.path.append("part_d_rag")

import os
import json
import torch
from common import *
from eurosat import patchify
from mae_model import MAE
from vae_model import kl_term
from schedule import T, alpha_bars, add_noise
from metrics import hit_at_k, recall_at_k, mrr_at_10

torch.manual_seed(0)
make_folders()

lines = []


def report(text):
    print(text)
    lines.append(text)


report("----- MAE -----")
model = MAE()
images = torch.rand(6, 3, 64, 64)
patches = patchify(images)
loss, pred, mask = model(patches)
masked_per_image = mask.sum(dim=1).tolist()
report("masked patches in each image: " + str(masked_per_image))
report("exactly 48 of 64 masked in every image: " + str(all(m == 48 for m in masked_per_image)))

changed = pred.detach().clone()
visible_position = (mask[0] == 0).nonzero()[0].item()
changed[0, visible_position] = changed[0, visible_position] + 100.0
loss_visible_changed = model.masked_loss(changed, patches, mask)
report("loss unchanged when a visible patch prediction is changed: " +
       str(torch.allclose(loss, loss_visible_changed)))

changed = pred.detach().clone()
masked_position = (mask[0] == 1).nonzero()[0].item()
changed[0, masked_position] = changed[0, masked_position] + 10.0
loss_masked_changed = model.masked_loss(changed, patches, mask)
report("loss changes when a masked patch prediction is changed: " +
       str(not torch.allclose(loss, loss_masked_changed)))

report("")
report("----- VAE -----")
zero = torch.zeros(5, 16)
report("KL when mu = 0 and logvar = 0: " + str(kl_term(zero, zero).item()))
smallest = 1000.0
for i in range(200):
    mu = torch.randn(64, 16) * 3
    logvar = torch.randn(64, 16) * 2
    each = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim=1)
    smallest = min(smallest, each.min().item())
report("smallest KL value over 12800 random cases: " + str(round(smallest, 6)) + " (never negative: " + str(smallest >= -1e-4) + ")")

report("")
report("----- DDPM -----")
report("alpha_bar at the last step: " + str(alpha_bars[T - 1].item()) + " (about 4e-5)")
x0 = torch.rand(1, 1, 28, 28) * 2 - 1
noise = torch.randn_like(x0)
x_last = add_noise(x0, torch.tensor([T - 1]), noise)
correlation_last = torch.corrcoef(torch.stack([x0.flatten(), x_last.flatten()]))[0, 1].item()
report("correlation between x0 and the noised image at t = 999: " + str(round(correlation_last, 4)) + " (close to 0, looks like noise)")
x_first = add_noise(x0, torch.tensor([0]), noise)
correlation_first = torch.corrcoef(torch.stack([x0.flatten(), x_first.flatten()]))[0, 1].item()
report("correlation between x0 and the noised image at t = 0: " + str(round(correlation_first, 4)) + " (close to 1)")

report("")
report("----- RAG -----")
all_good = True
for size in [1, 2, 5, 15]:
    relevant = list(range(100, 100 + size))
    ranked = relevant + list(range(500, 520))
    for k in [1, 5, 10]:
        if hit_at_k(ranked, relevant, k) != 1.0 or recall_at_k(ranked, relevant, k) != 1.0:
            all_good = False
    if mrr_at_10(ranked, relevant) != 1.0:
        all_good = False
report("perfect retriever gives Hit@k = Recall@k = MRR@10 = 1: " + str(all_good))

if os.path.exists("cache/chunks.json") and os.path.exists("cache/embeddings.pt"):
    file = open("cache/chunks.json", encoding="utf-8")
    data = json.load(file)
    file.close()
    longest = 0
    for text in data["texts"]:
        longest = max(longest, len(text.split()))
    report("longest chunk in words: " + str(longest) + " (at most 150: " + str(longest <= 150) + ")")
    embeddings = torch.load("cache/embeddings.pt")
    norms = embeddings.norm(dim=1)
    report("embedding norms: min " + str(round(norms.min().item(), 5)) + ", max " + str(round(norms.max().item(), 5)))
else:
    report("chunk and embedding checks skipped: run part_d_rag/build_index.py first")

file = open("results/correctness_checks.txt", "w")
file.write("\n".join(lines))
file.close()
print("saved results/correctness_checks.txt")
