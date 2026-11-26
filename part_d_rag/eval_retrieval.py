import sys
sys.path.append(".")
sys.path.append("starter")
sys.path.append("part_d_rag")

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from bioasq import load_corpus, load_questions
from rag import load_retriever, embed_texts, Index
from metrics import hit_at_k, recall_at_k, mrr_at_10

make_folders()

corpus = load_corpus()
questions = load_questions("test", corpus)
print("test questions:", len(questions))

tokenizer, model = load_retriever()
index = Index()

question_texts = []
for q in questions:
    question_texts.append(q["question"])
question_vectors = embed_texts(question_texts, tokenizer, model)

ranked_lists = []
for i in range(len(questions)):
    ranked_lists.append(index.top_abstracts(question_vectors[i], 20))
print("searched all questions")

table = {}
for k in [1, 5, 10]:
    hits = []
    recalls = []
    for i in range(len(questions)):
        hits.append(hit_at_k(ranked_lists[i], questions[i]["relevant_ids"], k))
        recalls.append(recall_at_k(ranked_lists[i], questions[i]["relevant_ids"], k))
    table["hit@" + str(k)] = mean(hits)
    table["recall@" + str(k)] = mean(recalls)

mrr_values = []
for i in range(len(questions)):
    mrr_values.append(mrr_at_10(ranked_lists[i], questions[i]["relevant_ids"]))
table["mrr@10"] = mean(mrr_values)

recall_curve = []
for k in range(1, 21):
    values = []
    for i in range(len(questions)):
        values.append(recall_at_k(ranked_lists[i], questions[i]["relevant_ids"], k))
    recall_curve.append(mean(values))

for name in table:
    print(name, round(table[name], 4))

failures = []
for i in range(len(questions)):
    if hit_at_k(ranked_lists[i], questions[i]["relevant_ids"], 10) == 0.0:
        relevant_id = questions[i]["relevant_ids"][0]
        top_id = ranked_lists[i][0]
        failures.append({"index": i, "question": questions[i]["question"], "reference_answer": questions[i]["answer"],
                         "relevant_id": relevant_id, "relevant_text": corpus[relevant_id][:400],
                         "top1_id": top_id, "top1_text": corpus[top_id][:400]})
print("questions with no relevant abstract in the top 10:", len(failures))

save_json({"table": table, "recall_curve": recall_curve, "num_questions": len(questions),
           "num_failures": len(failures)}, "results/retrieval.json")
save_json(failures[:15], "results/retrieval_failures.json")

plt.figure(figsize=(6, 4))
plt.plot(range(1, 21), recall_curve, marker="o")
plt.xlabel("k")
plt.ylabel("Recall@k")
plt.title("Retrieval Recall@k")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/fig_rag_recall.png", dpi=150)
print("saved results/fig_rag_recall.png")
