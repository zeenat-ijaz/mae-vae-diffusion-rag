import sys
sys.path.append(".")

import csv
from common import *

key = load_json("results/blind_test_key.json")

file = open("results/human_eval.csv", encoding="utf-8")
rows = list(csv.DictReader(file))
file.close()


def is_yes(text):
    return text.strip().lower() in ["y", "yes", "1", "true"]


correct1 = 0
correct2 = 0
agreed = 0
diffusion_rows = 0
empty = 0

for row in rows:
    if row["kind"] != "diffusion":
        continue
    diffusion_rows = diffusion_rows + 1
    truth = key[str(row["item"])].strip().lower()
    answer1 = row["member1_answer"].strip().lower()
    answer2 = row["member2_answer"].strip().lower()
    if answer1 == "" or answer2 == "":
        empty = empty + 1
    if answer1 == truth:
        correct1 = correct1 + 1
    if answer2 == truth:
        correct2 = correct2 + 1
    if answer1 == answer2:
        agreed = agreed + 1

supported1 = 0
supported2 = 0
right1 = 0
right2 = 0
rag_rows = 0
for row in rows:
    if row["kind"] != "rag":
        continue
    rag_rows = rag_rows + 1
    if is_yes(row["member1_supported"]):
        supported1 = supported1 + 1
    if is_yes(row["member2_supported"]):
        supported2 = supported2 + 1
    if is_yes(row["member1_correct"]):
        right1 = right1 + 1
    if is_yes(row["member2_correct"]):
        right2 = right2 + 1

print("diffusion blind test (" + str(diffusion_rows) + " images, " + str(empty) + " with a missing answer)")
print("member 1 correct:", correct1, "| member 2 correct:", correct2)
print("images where both members agreed:", agreed)
print("")
print("RAG human check (" + str(rag_rows) + " answers)")
print("supported, member 1 / member 2:", supported1, "/", supported2)
print("correct, member 1 / member 2:", right1, "/", right2)

save_json({"diffusion_images": diffusion_rows, "member1_correct": correct1, "member2_correct": correct2,
           "both_agreed": agreed, "rag_items": rag_rows,
           "supported_member1": supported1, "supported_member2": supported2,
           "correct_member1": right1, "correct_member2": right2}, "results/human_eval_scores.json")
print("saved results/human_eval_scores.json")
