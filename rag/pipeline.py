from rag.retriever import SimpleRetriever

documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Tokyo is the capital of Japan."
]

_retriever = None


def get_retriever():
    global _retriever

    if _retriever is None:
        _retriever = SimpleRetriever(documents)

    return _retriever

def retrieve(query, k=2):
    return get_retriever().retrieve(query, k=k)