from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Tokyo is the capital of Japan."
]

doc_embeddings = embedder.encode(documents)

index = faiss.IndexFlatL2(len(doc_embeddings[0]))
index.add(np.array(doc_embeddings))

def retrieve(query, k=2):
    q_emb = embedder.encode([query])
    distances, indices = index.search(np.array(q_emb), k)
    return [documents[i] for i in indices[0]]