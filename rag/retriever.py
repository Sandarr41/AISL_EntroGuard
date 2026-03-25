from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SimpleRetriever:
    def __init__(self, documents):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.documents = documents
        embeddings = self.embedder.encode(documents)

        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(np.array(embeddings))

    def retrieve(self, query, k=2):
        q_emb = self.embedder.encode([query])
        _, indices = self.index.search(np.array(q_emb), k)
        return [self.documents[i] for i in indices[0]]