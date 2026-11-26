import sys
sys.path.append(".")

import os
from common import *


def get(path):
    if os.path.exists(path):
        return load_json(path)
    return None


def r(value, digits=4):
    return str(round(value, digits))


mae = get("results/mae_pretrain.json")
clf = get("results/mae_classifier.json")
vae = get("results/vae_train.json")
diff = get("results/diffusion_train.json")
index_info = get("results/rag_index_info.json")
retrieval = get("results/retrieval.json")
generation = get("results/generation.json")
human = get("results/human_eval_scores.json")

out = []


def table(title, header, rows):
    out.append("## " + title)
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    out.append("")


rows = []
if mae and clf:
    rows.append(["EuroSAT train / test images; labelled images used",
                 str(mae["train_images"]) + " / " + str(clf["test_images"]) + "; " + str(clf["labelled_images"])])
if vae:
    rows.append(["UHaT classes; train / test images",
                 str(vae["classes"]) + "; " + str(vae["train_images"]) + " / " + str(vae["test_images"])])
if index_info:
    rows.append(["BioASQ abstracts; chunks; test questions",
                 str(index_info["abstracts"]) + "; " + str(index_info["chunks"]) + "; " + str(index_info["test_questions"])])
    rows.append(["Mean relevant abstracts per question", r(index_info["mean_relevant"], 2)])
table("Table 1 - Data", ["Item", "Value"], rows)

rows = []
if mae:
    rows.append(["MAE", str(mae["params"]), r(mae["epoch_losses"][-1]), r(mae["seconds"] / 60, 1) + " min / " + mae["gpu"]])
if vae:
    rows.append(["VAE", str(vae["params"]), r(vae["train_total"][-1], 2), r(vae["seconds"] / 60, 1) + " min / " + vae["gpu"]])
if diff:
    rows.append(["DDPM U-Net", str(diff["params"]), r(diff["epoch_losses"][-1]), r(diff["seconds"] / 60, 1) + " min / " + diff["gpu"]])
table("Table 2 - Training", ["Model", "Parameters", "Final loss", "Time / GPU"], rows)

rows = []
if clf:
    rows.append(["MAE classifier test accuracy", r(clf["test_accuracy"])])
if vae:
    rows.append(["VAE test loss = recon + KL",
                 r(vae["test_total"][-1], 2) + " = " + r(vae["test_recon"][-1], 2) + " + " + r(vae["test_kl"][-1], 2)])
if human:
    rows.append(["Diffusion blind test correct (member 1 / member 2, of 40)",
                 str(human["member1_correct"]) + " / " + str(human["member2_correct"])])
    rows.append(["Diffusion images both agreed on (of 40)", str(human["both_agreed"])])
table("Table 3 - Image results", ["Item", "Value"], rows)

if retrieval:
    t = retrieval["table"]
    names = ["hit@1", "hit@5", "hit@10", "recall@1", "recall@5", "recall@10", "mrr@10"]
    table("Table 4 - Retrieval", names, [[r(t[name]) for name in names]])

rows = []
if generation:
    rows.append(["ROUGE-L", r(generation["rouge_l"])])
    rows.append(["Citation precision", r(generation["citation_precision"])])
    rows.append(["I don't know rate", r(generation["dont_know_rate"])])
if human:
    rows.append(["Supported (member 1 / member 2, of 20)",
                 str(human["supported_member1"]) + " / " + str(human["supported_member2"])])
    rows.append(["Correct (member 1 / member 2, of 20)",
                 str(human["correct_member1"]) + " / " + str(human["correct_member2"])])
table("Table 5 - Generation", ["Item", "Value"], rows)

file = open("results/tables.md", "w", encoding="utf-8")
file.write("\n".join(out))
file.close()
for line in out:
    if not line.startswith("|---"):
        print(line)
