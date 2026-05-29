from collections import defaultdict

from datasets import load_dataset


BEIR_DATASETS = {
    "arguana": "BeIR/arguana",
    "fever": "BeIR/fever",
}


def _load_hf_dataset(path, name=None, split=None):
    if name is None:
        return load_dataset(path, split=split)
    return load_dataset(path, name, split=split)


def _load_beir_qrels(name, split):
    qrels_ds = load_dataset(f"BeIR/{name}-qrels", split=split)
    qrels = defaultdict(dict)
    for row in qrels_ds:
        query_id = row.get("query-id") or row.get("query_id") or row.get("qid")
        corpus_id = row.get("corpus-id") or row.get("corpus_id") or row.get("docid")
        score = row.get("score", 1)
        if query_id and corpus_id:
            qrels[str(query_id)][str(corpus_id)] = int(score)
    return qrels


def _load_beir_corpus(path, limit=None):
    corpus_ds = _load_hf_dataset(path, "corpus", split="corpus")
    corpus = {}
    for i, row in enumerate(corpus_ds):
        if limit and i >= limit:
            break
        doc_id = str(row.get("_id"))
        corpus[doc_id] = {
            "title": row.get("title", ""),
            "text": row.get("text", ""),
        }
    return corpus


def load_beir(name, split="test", limit=None, corpus_limit=None, include_corpus=False):
    """Load a BEIR retrieval dataset with queries and qrels.

    Set `include_corpus=True` to attach a corpus slice to each row. Keeping
    this disabled by default prevents huge BEIR corpora, especially FEVER,
    from being duplicated when rows are serialized by the generic evaluator.
    """
    if name not in BEIR_DATASETS:
        raise ValueError(f"Unsupported BEIR dataset: {name}")

    path = BEIR_DATASETS[name]
    queries_ds = _load_hf_dataset(path, "queries", split="queries")
    qrels = _load_beir_qrels(name, split)
    corpus = _load_beir_corpus(path, corpus_limit) if include_corpus else None

    dataset = []
    for i, row in enumerate(queries_ds):
        if limit and i >= limit:
            break

        query_id = str(row.get("_id"))
        query = row.get("text", "")
        relevant_docs = qrels.get(query_id, {})

        dataset.append({
            "prompt": query,
            "label": "safe",
            "category": f"retrieval/{name}",
            "task": "retrieval",
            "source_dataset": f"beir/{name}",
            "query_id": query_id,
            "query": query,
            "relevant_doc_ids": list(relevant_docs.keys()),
            "qrels": relevant_docs,
            "corpus": corpus or {},
        })

    return dataset


def load_beir_arguana(split="test", limit=None, corpus_limit=None, include_corpus=False):
    return load_beir(
        "arguana",
        split=split,
        limit=limit,
        corpus_limit=corpus_limit,
        include_corpus=include_corpus,
    )


def load_beir_fever(split="test", limit=None, corpus_limit=None, include_corpus=False):
    return load_beir(
        "fever",
        split=split,
        limit=limit,
        corpus_limit=corpus_limit,
        include_corpus=include_corpus,
    )