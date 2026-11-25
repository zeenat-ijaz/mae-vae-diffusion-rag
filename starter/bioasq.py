"""Part D data: biomedical questions + PubMed abstracts (BioASQ, mini version)."""
from datasets import load_dataset

BIOASQ_REPO = "enelpol/rag-mini-bioasq"

def load_corpus():
    """Returns {passage_id (int): passage text (str)} for all 40k abstracts."""
    ds = load_dataset(BIOASQ_REPO, "text-corpus")
    split = list(ds.keys())[0]             # the corpus has a single split
    return {int(r["id"]): r["passage"] for r in ds[split] if r["passage"]}

def load_questions(split="test", corpus=None):
    """Returns a list of dicts: {qid, question, answer, relevant_ids}.
    relevant_ids = the ids of the abstracts that answer the question (the
    ground truth for measuring retrieval). If `corpus` is given, ids that are
    not in the corpus are dropped, and questions left with none are skipped."""
    ds = load_dataset(BIOASQ_REPO, "question-answer-passages", split=split)
    out = []
    for r in ds:
        rel = [int(i) for i in r["relevant_passage_ids"]]
        if corpus is not None:
            rel = [i for i in rel if i in corpus]
        if rel:
            out.append({"qid": int(r["id"]), "question": r["question"],
                        "answer": r["answer"], "relevant_ids": rel})
    return out
