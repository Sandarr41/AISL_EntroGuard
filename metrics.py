import math
from collections import Counter
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu2(hyp_tokens, ref_tokens):
    if len(hyp_tokens) == 0:
        return 0.0
    precisions = []
    for n in (1, 2):
        hyp_ng, ref_ng = _ngrams(hyp_tokens, n), _ngrams(ref_tokens, n)
        overlap = sum((hyp_ng & ref_ng).values())
        total = max(sum(hyp_ng.values()), 1)
        precisions.append(overlap / total if total else 0.0)
    if min(precisions) == 0:
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))
    return bp * geo_mean


def rouge1_f1(hyp_tokens, ref_tokens):
    hyp_c, ref_c = Counter(hyp_tokens), Counter(ref_tokens)
    overlap = sum((hyp_c & ref_c).values())
    if overlap == 0:
        return 0.0
    p = overlap / max(sum(hyp_c.values()), 1)
    r = overlap / max(sum(ref_c.values()), 1)
    return 2 * p * r / (p + r)


def exact_match(hyp_tokens, ref_tokens):
    return float(hyp_tokens == ref_tokens)


def privacy_leakage_report(hyps, refs):
    """hyps/refs: list[list[str]] списки токенов. Ниже = лучше приватность."""
    n = len(hyps)
    bleu = sum(bleu2(h, r) for h, r in zip(hyps, refs)) / n
    rouge = sum(rouge1_f1(h, r) for h, r in zip(hyps, refs)) / n
    emr = sum(exact_match(h, r) for h, r in zip(hyps, refs)) / n
    return {"BLEU-2": bleu, "ROUGE-1": rouge, "EMR": emr}


@torch.no_grad()
def retrieval_recall_overlap(db_embeddings, orig_query_emb, protected_query_emb, k=5):
    """eq.(5): доля пересечения top-k выдачи для исходного и защищённого
    эмбеддинга запроса, относительно общей базы документов."""
    db = F.normalize(db_embeddings, dim=-1)
    q_orig = F.normalize(orig_query_emb, dim=-1)
    q_prot = F.normalize(protected_query_emb, dim=-1)

    sims_orig = q_orig @ db.T
    sims_prot = q_prot @ db.T
    topk_orig = sims_orig.topk(k, dim=-1).indices
    topk_prot = sims_prot.topk(k, dim=-1).indices

    overlaps = []
    for i in range(topk_orig.size(0)):
        so, sp = set(topk_orig[i].tolist()), set(topk_prot[i].tolist())
        overlaps.append(len(so & sp) / k)
    return sum(overlaps) / len(overlaps)
