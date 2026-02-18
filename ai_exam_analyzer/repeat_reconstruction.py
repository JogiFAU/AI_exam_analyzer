"""Detect repeated questions across years and derive reconstruction suggestions."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class RepeatSuggestion:
    cluster_id: int
    anchor_question_id: str
    confidence: float
    suggested_correct_indices: List[int]
    matched_correct_texts: List[str]


def _norm_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _tokenize(question: Dict[str, Any]) -> Set[str]:
    parts = [str(question.get("questionText") or ""), str(question.get("explanationText") or "")]
    for a in question.get("answers") or []:
        parts.append(str(a.get("text") or ""))
    out: Set[str] = set()
    for raw in "\n".join(parts).lower().replace("\n", " ").split():
        tok = "".join(ch for ch in raw if ch.isalnum() or ch in "äöüß")
        if len(tok) >= 3:
            out.add(tok)
    return out


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _candidate_pairs(items: List[Set[str]]) -> Set[Tuple[int, int]]:
    inv: Dict[str, List[int]] = defaultdict(list)
    for i, toks in enumerate(items):
        for t in toks:
            inv[t].append(i)
    pairs: Set[Tuple[int, int]] = set()
    for idxs in inv.values():
        if len(idxs) <= 1:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                pairs.add((a, b) if a < b else (b, a))
    return pairs


def _similarity(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _question_year(q: Dict[str, Any]) -> str:
    y = q.get("examYear")
    return str(y) if y is not None else ""


def _question_conf(q: Dict[str, Any]) -> float:
    audit = q.get("aiAudit") or {}
    return float((((audit.get("answerPlausibility") or {}).get("finalCombinedConfidence")) or 0.0))


def _is_high_quality_anchor(q: Dict[str, Any], min_conf: float) -> bool:
    audit = q.get("aiAudit") or {}
    if audit.get("status") != "completed":
        return False
    maintenance = audit.get("maintenance") or {}
    if bool(maintenance.get("needsMaintenance")):
        return False
    if _question_conf(q) < min_conf:
        return False
    if not (q.get("correctIndices") or []):
        return False
    return True


def _anchor_correct_texts(q: Dict[str, Any]) -> List[str]:
    by_correct_answers = q.get("correctAnswers") or []
    if by_correct_answers:
        texts = [_norm_text(x.get("text", "")) for x in by_correct_answers if _norm_text(x.get("text", ""))]
        if texts:
            return texts

    texts: List[str] = []
    for a in q.get("answers") or []:
        if bool(a.get("isCorrect")):
            t = _norm_text(a.get("text", ""))
            if t:
                texts.append(t)
    return texts


def _derive_external_indices(q: Dict[str, Any]) -> List[int]:
    out: List[int] = []
    for i, a in enumerate(q.get("answers") or []):
        idx = None
        for key in ("answerIndex", "position", "index"):
            value = a.get(key)
            if isinstance(value, int) and value > 0:
                idx = value
                break
        out.append(idx if idx is not None else (i + 1))
    return out


def _map_anchor_texts_to_target_indices(target: Dict[str, Any], anchor_correct_texts: List[str]) -> List[int]:
    if not anchor_correct_texts:
        return []
    wanted = set(anchor_correct_texts)
    indices = _derive_external_indices(target)
    out: List[int] = []
    for i, a in enumerate(target.get("answers") or []):
        t = _norm_text(a.get("text", ""))
        if t and t in wanted and i < len(indices):
            out.append(indices[i])
    return sorted(set(out))


def compute_repeat_reconstruction(
    questions: List[Dict[str, Any]],
    *,
    min_similarity: float,
    min_anchor_conf: float,
) -> Tuple[Dict[str, RepeatSuggestion], Dict[str, Any]]:
    """Return suggestions for low-quality repeated questions and summary metrics."""
    toks = [_tokenize(q) for q in questions]
    uf = _UnionFind(len(questions))
    for i, j in _candidate_pairs(toks):
        if _similarity(toks[i], toks[j]) >= min_similarity:
            uf.union(i, j)

    root_to_members: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(questions)):
        root_to_members[uf.find(i)].append(i)

    qid_to_suggestion: Dict[str, RepeatSuggestion] = {}
    clusters_considered = 0
    clusters_cross_year = 0

    for cluster_idx, members in enumerate(root_to_members.values(), start=1):
        if len(members) <= 1:
            continue
        clusters_considered += 1

        years = {_question_year(questions[i]) for i in members if _question_year(questions[i])}
        if len(years) < 2:
            continue
        clusters_cross_year += 1

        anchors = [i for i in members if _is_high_quality_anchor(questions[i], min_anchor_conf)]
        if not anchors:
            continue

        anchors.sort(key=lambda idx: _question_conf(questions[idx]), reverse=True)
        best_anchor_idx = anchors[0]
        anchor_q = questions[best_anchor_idx]
        anchor_id = str(anchor_q.get("id") or "")
        anchor_conf = _question_conf(anchor_q)
        anchor_texts = _anchor_correct_texts(anchor_q)

        for m in members:
            if m == best_anchor_idx:
                continue
            target = questions[m]
            target_id = str(target.get("id") or "")
            if not target_id:
                continue

            maintenance = ((target.get("aiAudit") or {}).get("maintenance") or {})
            target_low_quality = bool(maintenance.get("needsMaintenance")) or _question_conf(target) < min_anchor_conf
            if not target_low_quality:
                continue

            suggested_indices = _map_anchor_texts_to_target_indices(target, anchor_texts)
            if not suggested_indices:
                continue

            qid_to_suggestion[target_id] = RepeatSuggestion(
                cluster_id=cluster_idx,
                anchor_question_id=anchor_id,
                confidence=round(anchor_conf, 4),
                suggested_correct_indices=suggested_indices,
                matched_correct_texts=anchor_texts,
            )

    summary = {
        "clustersConsidered": clusters_considered,
        "crossYearClusters": clusters_cross_year,
        "suggestions": len(qid_to_suggestion),
    }
    return qid_to_suggestion, summary
