import sys
sys.path.append(".")
sys.path.append("starter")
sys.path.append("part_d_rag")

import json
import time
import torch
from common import *
from bioasq import load_corpus, load_questions
from rag import split_into_chunks, load_retriever, embed_texts

make_folders()

corpus = load_corpus()
questions = load_questions("test", corpus)
print("abstracts:", len(corpus), "| test questions:", len(questions))

chunk_abstract_ids = []
chunk_texts = []
for abstract_id in sorted(corpus.keys()):
    for piece in split_into_chunks(corpus[abstract_id]):
        chunk_abstract_ids.append(abstract_id)
        chunk_texts.append(piece)
print("chunks:", len(chunk_texts))

longest = 0
for piece in chunk_texts:
    longest = max(longest, len(piece.split()))
print("longest chunk in words:", longest)

tokenizer, model = load_retriever()
start_time = time.time()
embeddings = embed_texts(chunk_texts, tokenizer, model)
seconds = time.time() - start_time
print("embedding matrix:", tuple(embeddings.shape), "| time in minutes:", round(seconds / 60, 1))

torch.save(embeddings, "cache/embeddings.pt")
file = open("cache/chunks.json", "w", encoding="utf-8")
json.dump({"abstract_ids": chunk_abstract_ids, "texts": chunk_texts}, file)
file.close()

total_relevant = 0
for q in questions:
    total_relevant = total_relevant + len(set(q["relevant_ids"]))

save_json({"abstracts": len(corpus), "chunks": len(chunk_texts), "test_questions": len(questions),
           "mean_relevant": total_relevant / len(questions), "embedding_seconds": seconds,
           "gpu": gpu_name()}, "results/rag_index_info.json")
print("saved cache/embeddings.pt, cache/chunks.json and results/rag_index_info.json")
