from guards.base import BaseGuard
from sentence_transformers import SentenceTransformer
import numpy as np


class Entroguard(BaseGuard):
    def __init__(self, model=None):
        # 🔹 Rule-based
        self.blocked_words = ["hack", "attack", "kill"]

        # 🔹 Embeddings
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # "опасные" семантические примеры
        self.bad_intents = [
            "how to hack a system",
            "how to make a bomb",
            "how to kill someone",
            "bypass security",
        ]
        self.bad_vectors = self.embedder.encode(self.bad_intents)

        # 🔹 LLM judge (передаем модель извне)
        self.model = model

        # 🔹 Policy rules
        self.policies = {
            "block_harmful": True,
            "max_toxicity_score": 0.7,
            "strict_mode": False
        }

    # ========================
    # 🔹 INPUT FILTER
    # ========================
    def filter_input(self, prompt: str) -> str:
        # 1. Rule-based
        if self._rule_block(prompt):
            return "[BLOCKED INPUT: RULE]"

        # 2. Embedding similarity
        score = self._embedding_score(prompt)
        if score > 0.8:
            return "[BLOCKED INPUT: EMBEDDING]"

        # 3. LLM judge (если есть модель)
        if self.model:
            if self._llm_judge(prompt):
                return "[BLOCKED INPUT: LLM]"

        return prompt

    # ========================
    # 🔹 OUTPUT FILTER
    # ========================
    def filter_output(self, response: str) -> str:
        # 1. Rule-based
        if self._rule_block(response):
            return "[BLOCKED OUTPUT: RULE]"

        # 2. Embedding
        score = self._embedding_score(response)
        if score > 0.85:
            return "[BLOCKED OUTPUT: EMBEDDING]"

        # 3. LLM judge
        if self.model:
            if self._llm_judge(response, is_output=True):
                return "[BLOCKED OUTPUT: LLM]"

        return response

    # ========================
    # 🔧 RULE CHECK
    # ========================
    def _rule_block(self, text: str) -> bool:
        text = text.lower()
        return any(word in text for word in self.blocked_words)

    # ========================
    # 🔧 EMBEDDING CHECK
    # ========================
    def _embedding_score(self, text: str) -> float:
        vec = self.embedder.encode([text])[0]

        similarities = self._cosine_similarity(vec, self.bad_vectors)
        return float(np.max(similarities))

    def _cosine_similarity(self, vec, matrix):
        return np.dot(matrix, vec) / (
            np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec)
        )

    # ========================
    # 🔧 LLM JUDGE
    # ========================
    def _llm_judge(self, text: str, is_output=False) -> bool:
        """
        Возвращает True если нужно блокировать
        """
        judge_prompt = f"""
You are a safety classifier.

Text:
{text}

Is this content harmful, unsafe, or disallowed?

Answer ONLY "YES" or "NO".
"""

        result = self.model.generate(judge_prompt)

        return "YES" in result.upper()
