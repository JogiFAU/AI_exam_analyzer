"""Deterministic topic-candidate retrieval for constrained LLM topic assignment."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in (text or "").lower().replace("\n", " ").split():
        token = "".join(ch for ch in raw if ch.isalnum() or ch in "äöüß")
        if len(token) >= 3:
            tokens.append(token)
    return tokens


class TopicCandidateIndex:
    def __init__(self, catalog: List[Dict[str, Any]]):
        self.catalog = catalog
        self.df: Dict[str, int] = defaultdict(int)
        self.docs: Dict[str, Counter[str]] = {}
        self._build()

    def _build(self) -> None:
        for row in self.catalog:
            text = f"{row.get('superTopicName','')} {row.get('subtopicName','')}"
            toks = _tokenize(text)
            counts = Counter(toks)
            key = str(row.get("topicKey") or "")
            self.docs[key] = counts
            for tok in set(toks):
                self.df[tok] += 1

    def _idf(self, token: str) -> float:
        n_docs = max(1, len(self.docs))
        df = self.df.get(token, 0)
        return math.log((1 + n_docs) / (1 + df)) + 1.0

    def rank(self, question: Dict[str, Any], *, top_k: int = 3) -> List[Dict[str, Any]]:
        parts = [question.get("questionText", "")]
        for ans in question.get("answers") or []:
            parts.append(ans.get("text", ""))
        if question.get("explanationText"):
            parts.append(question.get("explanationText", ""))

        q_counts = Counter(_tokenize("\n".join(parts)))
        if not q_counts:
            return []

        scored: List[Dict[str, Any]] = []
        for row in self.catalog:
            topic_key = str(row.get("topicKey") or "")
            d_counts = self.docs.get(topic_key, Counter())
            shared: Set[str] = set(q_counts) & set(d_counts)
            if not shared:
                continue
            score = 0.0
            for tok in shared:
                score += min(q_counts[tok], d_counts[tok]) * self._idf(tok)
            scored.append({
                "topicKey": topic_key,
                "superTopic": row.get("superTopicName", ""),
                "subtopic": row.get("subtopicName", ""),
                "score": round(score, 4),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: max(1, int(top_k))]
