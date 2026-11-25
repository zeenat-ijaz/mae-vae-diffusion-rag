import os
import torch
from viz import save_grid
from eurosat import load_eurosat_split, few_label_indices, patchify, unpatchify, CLASS_NAMES
from uhat import load_uhat_characters
from bioasq import load_corpus, load_questions

os.makedirs("results", exist_ok=True)

# ---- Part A: EuroSAT ----
x_tr, y_tr = load_eurosat_split("train")
x_te, y_te = load_eurosat_split("test")
few = few_label_indices(y_tr)
p = patchify(x_tr[:8])
print("EuroSAT train", tuple(x_tr.shape), "test", tuple(x_te.shape))
print("few-label subset:", len(few), "images; patches per image:", tuple(p.shape))
assert torch.equal(unpatchify(p), x_tr[:8])
save_grid(x_tr[:32], "results/eurosat_examples.png", nrow=8,
          titles=[CLASS_NAMES[i] for i in y_tr[:32]])

# ---- Parts B and C: UHaT characters ----
c_tr, l_tr, names = load_uhat_characters(split="train")
c_te, l_te, _ = load_uhat_characters(split="test", class_names=names)
print("UHaT train", tuple(c_tr.shape), "test", tuple(c_te.shape), "classes:", len(names))
print("class names:", names)
pick = torch.randperm(len(c_tr))[:64]
save_grid(c_tr[pick], "results/uhat_examples.png", nrow=8, titles=[names[i] for i in l_tr[pick]])

# ---- Part D: BioASQ ----
corpus = load_corpus()
qs = load_questions("test", corpus)
print("BioASQ corpus:", len(corpus), "abstracts; test questions:", len(qs))
print("example:", qs[0]["question"], "| relevant ids:", qs[0]["relevant_ids"][:5])
