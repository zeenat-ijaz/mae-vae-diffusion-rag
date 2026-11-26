def hit_at_k(ranked, relevant, k):
    found = set(ranked[:k]) & set(relevant)
    if len(found) > 0:
        return 1.0
    return 0.0


def recall_at_k(ranked, relevant, k):
    relevant = set(relevant)
    found = set(ranked[:k]) & relevant
    return len(found) / min(k, len(relevant))


def mrr_at_10(ranked, relevant):
    relevant = set(relevant)
    for position, abstract_id in enumerate(ranked[:10]):
        if abstract_id in relevant:
            return 1.0 / (position + 1)
    return 0.0
