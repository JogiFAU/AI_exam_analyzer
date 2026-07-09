"""Deterministic topic-candidate retrieval for constrained LLM topic assignment."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set


STOPWORDS = {
    "aber", "alle", "als", "also", "am", "an", "auch", "auf", "aus", "bei", "der", "die", "das", "dem", "den",
    "ein", "eine", "einer", "einem", "einen", "für", "im", "in", "ist", "mit", "nach", "nicht", "oder", "und", "von", "zu",
    "frage", "abfrage", "aussage", "antwort", "antworten", "welche", "welcher", "welches", "trifft", "stimmt", "korrekt",
}


TOPIC_HINTS = {
    "respiratorische infekte / tuberkulose": "respiratorisch infekt tuberkulose pneumonie pneumonia bronchitis influenza pneumokokken streptococcus pneumoniae mykobakterien mycobacterium",
    "sepsis & meningitis": "sepsis meningitis liquor blutkultur meningokokken pneumokokken streptococcus pneumoniae neisseria meningitidis",
    "mikroskopie": "mikroskopie gram faerbung gramfärbung grampräparat morphologie mikroskopisch",
    "virushepatitis": "hepatitis hbv hcv hav hdv hev leber",
    "hiv / retroviren": "hiv retrovirus aids cd4 antiretroviral",
    "respiratorische viren": "respiratorisch influenza rsv corona sars cov parainfluenza adenovirus",
    "infektionsschutzgesetz / meldewesen": "ifsg meldepflicht meldewesen meldepflichtig gesundheitsamt",
    "vakzinologie (impfungen)": "impfung impfstoff vakzin vakzinologie immunisierung",
}

SYNONYMS = {
    "grampräparat": "gramfaerbung",
    "grampraeparat": "gramfaerbung",
    "gramfärbung": "gramfaerbung",
    "gramfaerbung": "gramfaerbung",
    "morphologisch": "morphologie",
    "morphologi": "morphologie",
    "pneumokokken": "pneumokokk",
    "pneumokokkus": "pneumokokk",
    "streptococcus": "pneumokokk",
    "pneumoniae": "pneumokokk",
    "isolationsmaßnahmen": "isolation",
    "isolationsmassnahmen": "isolation",
    "schutzmasken": "schutzmaske",
}


def _normalize_token(token: str) -> str:
    replacements = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}
    for src, dst in replacements.items():
        token = token.replace(src, dst)
    if token in SYNONYMS:
        return SYNONYMS[token]
    for suffix in ("ungen", "heiten", "keiten", "ischer", "liche", "lichen", "igkeit", "ionen"):
        if len(token) > len(suffix) + 4 and token.endswith(suffix):
            token = token[: -len(suffix)]
            return SYNONYMS.get(token, token)
    for suffix in ("ern", "er", "en", "es", "s"):
        if len(token) > len(suffix) + 5 and token.endswith(suffix):
            token = token[: -len(suffix)]
            return SYNONYMS.get(token, token)
    return SYNONYMS.get(token, token)


def _tokenize(text: str) -> List[str]:
    base_tokens: List[str] = []
    for raw in (text or "").lower().replace("\n", " ").split():
        token = "".join(ch for ch in raw if ch.isalnum() or ch in "äöüß")
        token = _normalize_token(token)
        if len(token) >= 3 and token not in STOPWORDS:
            base_tokens.append(token)

    tokens = list(base_tokens)
    for left, right in zip(base_tokens, base_tokens[1:]):
        if left != right:
            tokens.append(f"{left}_{right}")
    return tokens


class TopicCandidateIndex:
    def __init__(self, catalog: List[Dict[str, Any]]):
        self.catalog = catalog
        self.df: Dict[str, int] = defaultdict(int)
        self.docs: Dict[str, Counter[str]] = {}
        self.super_docs: Dict[str, Counter[str]] = {}
        self._build()

    def _build(self) -> None:
        for row in self.catalog:
            aliases = " ".join(row.get("aliases") or [])
            subtopic_name = str(row.get("subtopicName", ""))
            topic_hints = TOPIC_HINTS.get(subtopic_name.strip().lower(), "")
            # Weight the concrete subtopic/aliases more strongly than the broad
            # super-topic so generic domain terms do not dominate retrieval.
            text = " ".join([
                str(row.get("superTopicName", "")),
                str(row.get("subtopicName", "")),
                str(row.get("subtopicName", "")),
                aliases,
                aliases,
                topic_hints,
            ])
            toks = _tokenize(text)
            counts = Counter(toks)
            key = str(row.get("topicKey") or "")
            super_key = str(row.get("superTopicId") or "")
            self.docs[key] = counts
            self.super_docs.setdefault(super_key, Counter()).update(counts)
            for tok in set(toks):
                self.df[tok] += 1

    def _idf(self, token: str) -> float:
        n_docs = max(1, len(self.docs))
        df = self.df.get(token, 0)
        return math.log((1 + n_docs) / (1 + df)) + 1.0

    def rank(self, question: Dict[str, Any], *, top_k: int = 5) -> List[Dict[str, Any]]:
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
            matched_tokens: List[str] = []
            for tok in shared:
                token_score = min(q_counts[tok], d_counts[tok]) * self._idf(tok)
                if "_" in tok:
                    token_score *= 1.35
                score += token_score
                matched_tokens.append(tok)

            super_key = str(row.get("superTopicId") or "")
            super_shared = set(q_counts) & set(self.super_docs.get(super_key, Counter()))
            super_score = sum(self._idf(tok) for tok in super_shared) * 0.15
            score += super_score

            scored.append({
                "topicKey": topic_key,
                "superTopic": row.get("superTopicName", ""),
                "subtopic": row.get("subtopicName", ""),
                "score": round(score, 4),
                "matchedTokens": sorted(matched_tokens)[:12],
                "matchedTokenCount": len(matched_tokens),
            })

        scored.sort(key=lambda x: (x["score"], x["matchedTokenCount"]), reverse=True)
        top = scored[: max(1, int(top_k))]
        max_score = top[0]["score"] if top else 0.0
        second_score = top[1]["score"] if len(top) > 1 else 0.0
        for idx, row in enumerate(top):
            row["relativeScore"] = round((row["score"] / max_score), 4) if max_score > 0 else 0.0
            row["rank"] = idx + 1
            if idx == 0:
                row["marginToNext"] = round(max_score - second_score, 4)
        return top
