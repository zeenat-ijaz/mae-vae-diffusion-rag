import sys
sys.path.append(".")
sys.path.append("starter")
sys.path.append("part_d_rag")

import csv
import json
import random
import re
import os
import torch
from rouge_score import rouge_scorer
from common import *
from bioasq import load_corpus, load_questions
from rag import load_retriever, embed_texts, Index
from generate import load_generator, generate_answer

make_folders()

NUM_QUESTIONS = 100
PASSAGES = 5

corpus = load_corpus()
questions = load_questions("test", corpus)[:NUM_QUESTIONS]

retriever_tokenizer, retriever_model = load_retriever()
index = Index()
generator_tokenizer, generator_model = load_generator()

question_vectors = embed_texts([q["question"] for q in questions], retriever_tokenizer, retriever_model)
scorer = rouge_scorer.RougeScorer(["rougeL"])

dont_know_phrases = ["do not know", "don't know", "does not contain", "do not contain",
                     "not enough information", "cannot answer", "can't answer", "no information"]


def find_cited_numbers(answer):
    numbers = []
    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        for part in group.split(","):
            part = part.strip()
            if part.isdigit():
                numbers.append(int(part))
    return numbers


records = []
rouge_values = []
cited_total = 0
cited_correct = 0
dont_know_count = 0

for i in range(len(questions)):
    q = questions[i]
    chunk_ids = index.top_chunks(question_vectors[i], PASSAGES)
    passage_texts = []
    passages = []
    for number, chunk_id in enumerate(chunk_ids):
        passage_texts.append(index.chunk_texts[chunk_id])
        passages.append({"number": number + 1, "abstract_id": index.chunk_abstract_ids[chunk_id],
                         "text": index.chunk_texts[chunk_id]})

    answer = generate_answer(q["question"], passage_texts, generator_tokenizer, generator_model)

    rouge_values.append(scorer.score(q["answer"], answer)["rougeL"].fmeasure)

    cited = find_cited_numbers(answer)
    for number in cited:
        if 1 <= number <= PASSAGES:
            cited_total = cited_total + 1
            if passages[number - 1]["abstract_id"] in q["relevant_ids"]:
                cited_correct = cited_correct + 1

    said_dont_know = False
    for phrase in dont_know_phrases:
        if phrase in answer.lower():
            said_dont_know = True
    if said_dont_know:
        dont_know_count = dont_know_count + 1

    top10 = index.top_abstracts(question_vectors[i], 10)
    retrieved = False
    for abstract_id in top10:
        if abstract_id in q["relevant_ids"]:
            retrieved = True

    records.append({"index": i, "qid": q["qid"], "question": q["question"], "reference_answer": q["answer"],
                    "generated_answer": answer, "cited": cited, "passages": passages,
                    "relevant_ids": q["relevant_ids"], "relevant_in_top10": retrieved,
                    "said_dont_know": said_dont_know})
    print("answered", i + 1, "of", len(questions))

file = open("results/rag_answers.jsonl", "w", encoding="utf-8")
for record in records:
    file.write(json.dumps(record) + "\n")
file.close()

if cited_total > 0:
    citation_precision = cited_correct / cited_total
else:
    citation_precision = 0.0

metrics = {"rouge_l": mean(rouge_values), "citation_precision": citation_precision,
           "cited_total": cited_total, "dont_know_rate": dont_know_count / len(records),
           "questions": len(records)}
save_json(metrics, "results/generation.json")
print(metrics)

chosen = random.Random(0).sample(range(len(records)), 20)
chosen.sort()

lines = ["# RAG answers to be marked by both members", "",
         "For each item write supported (yes/no) and correct (yes/no) in results/human_eval.csv.", ""]
for item in chosen:
    r = records[item]
    lines.append("## Item " + str(item))
    lines.append("")
    lines.append("Question: " + r["question"])
    lines.append("")
    lines.append("Answer: " + r["generated_answer"])
    lines.append("")
    lines.append("Reference answer: " + r["reference_answer"])
    lines.append("")
    for p in r["passages"]:
        lines.append("[" + str(p["number"]) + "] (abstract " + str(p["abstract_id"]) + ") " + p["text"])
        lines.append("")
file = open("results/human_eval_rag_sheet.md", "w", encoding="utf-8")
file.write("\n".join(lines))
file.close()

columns = ["kind", "item", "text", "member1_answer", "member2_answer",
           "member1_supported", "member2_supported", "member1_correct", "member2_correct"]
old_rows = []
if os.path.exists("results/human_eval.csv"):
    file = open("results/human_eval.csv", encoding="utf-8")
    for row in csv.DictReader(file):
        if row["kind"] != "rag":
            old_rows.append(row)
    file.close()

file = open("results/human_eval.csv", "w", newline="", encoding="utf-8")
writer = csv.DictWriter(file, fieldnames=columns)
writer.writeheader()
for row in old_rows:
    writer.writerow(row)
for item in chosen:
    row = {}
    for c in columns:
        row[c] = ""
    row["kind"] = "rag"
    row["item"] = item
    row["text"] = records[item]["question"]
    writer.writerow(row)
file.close()
print("saved results/rag_answers.jsonl, results/human_eval_rag_sheet.md and rag rows in results/human_eval.csv")

failed = []
unsure = []
for r in records:
    if not r["relevant_in_top10"]:
        failed.append(r)
    if r["said_dont_know"]:
        unsure.append(r)

wanted = []
if len(failed) > 0:
    wanted.append(failed[0])
if len(unsure) > 0 and unsure[0] not in wanted:
    wanted.append(unsure[0])
for r in records:
    if len(wanted) < 5 and r not in wanted:
        wanted.append(r)

lines = ["# Qualitative samples", ""]
for r in wanted:
    lines.append("## Question " + str(r["index"]))
    lines.append("")
    lines.append("Question: " + r["question"])
    lines.append("")
    lines.append("Answer: " + r["generated_answer"])
    lines.append("")
    lines.append("Reference answer: " + r["reference_answer"])
    lines.append("")
    lines.append("Relevant abstract found in top 10: " + str(r["relevant_in_top10"]) +
                 " | model said it did not know: " + str(r["said_dont_know"]))
    lines.append("")
    for p in r["passages"]:
        lines.append("[" + str(p["number"]) + "] (abstract " + str(p["abstract_id"]) + ") " + p["text"][:300])
        lines.append("")
file = open("results/samples.md", "w", encoding="utf-8")
file.write("\n".join(lines))
file.close()
print("saved results/samples.md")
