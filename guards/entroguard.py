import re

import numpy as np
from sentence_transformers import SentenceTransformer

from guards.base import BaseGuard


class Entroguard(BaseGuard):
    def __init__(self, model=None):
        # 🔹 Rule-based
        self.blocked_words = ["hack", "attack", "kill"]
        self._blocked_pattern = re.compile(
            r"\b(" + "|".join(re.escape(word) for word in self.blocked_words) + r")\b",
            re.IGNORECASE,
        )

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

    # INPUT FILTER
        if self._llm_judge(prompt):
            return "[BLOCKED INPUT: LLM]"
        return prompt

    # OUTPUT FILTER
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

    def is_blocked(self, text: str) -> bool:
        return text.startswith("[BLOCKED")

    # RULE CHECK
    def _rule_block(self, text: str) -> bool:
        return bool(self._blocked_pattern.search(text))


    # EMBEDDING CHECK

    def _embedding_score(self, text: str) -> float:
        vec = self.embedder.encode([text])[0]

        similarities = self._cosine_similarity(vec, self.bad_vectors)
        return float(np.max(similarities))

    def _cosine_similarity(self, vec, matrix):
        vec_norm = np.linalg.norm(vec)
        matrix_norms = np.linalg.norm(matrix, axis=1)

        if vec_norm == 0:
            return np.zeros(matrix.shape[0])

        denominator = matrix_norms * vec_norm
        return np.divide(
            np.dot(matrix, vec),
            denominator,
            out=np.zeros_like(matrix_norms, dtype=float),
            where=denominator != 0,
        )

    # LLM JUDGE
    def _llm_judge(self, text: str, is_output=False) -> bool:
        """
        Возвращает True если нужно блокировать
        """
        content_type = "model output" if is_output else "user input"
        judge_prompt = f"""
You are a safety classifier.

Classify the following {content_type} as harmful, unsafe, or disallowed.

Text:
{text}

Answer ONLY "YES" or "NO".
"""

        result = self.model.generate(judge_prompt, max_new_tokens=3)
        decisions = re.findall(r"\b(YES|NO)\b", result.upper())

        return bool(decisions and decisions[-1] == "YES")