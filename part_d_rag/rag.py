import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

RETRIEVER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CHUNK_WORDS = 150
OVERLAP_WORDS = 30


def split_into_chunks(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    step = size - overlap
    chunks = []
    start = 0
    while True:
        piece = words[start:start + size]
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
        start = start + step
    return chunks


def load_retriever():
    tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_NAME)
    model = AutoModel.from_pretrained(RETRIEVER_NAME)
    model = model.to(device)
    model.eval()
    return tokenizer, model


def embed_texts(texts, tokenizer, model, batch_size=128):
    all_vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            encoded = encoded.to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            pooled = F.normalize(pooled, p=2, dim=1)
            all_vectors.append(pooled.cpu())
    return torch.cat(all_vectors, dim=0)


class Index:
    def __init__(self, folder="cache"):
        self.embeddings = torch.load(folder + "/embeddings.pt").to(device)
        file = open(folder + "/chunks.json", encoding="utf-8")
        data = json.load(file)
        file.close()
        self.chunk_abstract_ids = data["abstract_ids"]
        self.chunk_texts = data["texts"]

        self.abstract_ids = sorted(set(self.chunk_abstract_ids))
        position = {}
        for i, abstract_id in enumerate(self.abstract_ids):
            position[abstract_id] = i
        mapped = []
        for abstract_id in self.chunk_abstract_ids:
            mapped.append(position[abstract_id])
        self.chunk_to_abstract = torch.tensor(mapped, device=device)

    def top_abstracts(self, query_vector, k):
        scores = self.embeddings @ query_vector.to(device)
        best = torch.full((len(self.abstract_ids),), -10.0, device=device)
        best = best.scatter_reduce(0, self.chunk_to_abstract, scores, reduce="amax", include_self=True)
        top = best.topk(k).indices.tolist()
        return [self.abstract_ids[i] for i in top]

    def top_chunks(self, query_vector, k):
        scores = self.embeddings @ query_vector.to(device)
        top = scores.topk(k).indices.tolist()
        return top
