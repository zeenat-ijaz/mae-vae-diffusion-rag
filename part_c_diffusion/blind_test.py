import sys
sys.path.append(".")
sys.path.append("starter")
sys.path.append("part_c_diffusion")

import csv
import random
import torch
from common import *
from viz import save_grid
from unet import make_unet
from sample import sample_images, to_01

torch.manual_seed(7)
random.seed(7)
make_folders()

x_train, y_train, x_test, y_test, names = load_uhat()

unet = make_unet().to(device)
unet.load_state_dict(torch.load("weights/unet.pt", map_location=device))
unet.eval()

labels = torch.arange(40).to(device)
samples, saved = sample_images(unet, labels)
samples = to_01(samples).cpu()

order = list(range(40))
random.shuffle(order)

shuffled = samples[order]
save_grid(shuffled, "results/blind_test_images.png", nrow=8, titles=[str(i + 1) for i in range(40)])
print("saved results/blind_test_images.png (images numbered 1 to 40)")

key = {}
for position in range(40):
    key[str(position + 1)] = names[order[position]]
save_json(key, "results/blind_test_key.json")

columns = ["kind", "item", "text", "member1_answer", "member2_answer",
           "member1_supported", "member2_supported", "member1_correct", "member2_correct"]

file = open("results/human_eval.csv", "w", newline="", encoding="utf-8")
writer = csv.DictWriter(file, fieldnames=columns)
writer.writeheader()
for position in range(40):
    row = {}
    for c in columns:
        row[c] = ""
    row["kind"] = "diffusion"
    row["item"] = position + 1
    writer.writerow(row)
file.close()

print("saved results/human_eval.csv")
print("each member writes the character name for every image in member1_answer / member2_answer")
print("class names:", names)
print("do not open results/blind_test_key.json until both members have finished")
