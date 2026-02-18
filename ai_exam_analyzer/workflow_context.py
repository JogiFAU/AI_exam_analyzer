"""Dataset-level context enrichment (text/image clusters and abstractions)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


STOPWORDS_DE = {
    "aber", "alle", "als", "also", "am", "an", "auch", "auf", "aus", "bei", "der", "die", "das", "dem", "den",
    "ein", "eine", "einer", "einem", "einen", "für", "im", "in", "ist", "mit", "nicht", "oder", "und", "von", "zu",
}

TEMPLATE_TOKENS = {
    "was", "welche", "welcher", "welches", "aussage", "stimmt", "trifft", "ehesten", "richtig", "falsch", "liegt", "vor",
}


def _tokenize(text: str) -> Set[str]:
    out: Set[str] = set()
    for raw in (text or "").lower().replace("\n", " ").split():
        token = "".join(ch for ch in raw if ch.isalnum() or ch in "äöüß")
        if len(token) < 3:
            continue
        if token in STOPWORDS_DE or token in TEMPLATE_TOKENS:
            continue
        out.add(token)
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
    inverted: Dict[str, List[int]] = defaultdict(list)
    for idx, toks in enumerate(items):
        for t in toks:
            inverted[t].append(idx)

    pairs: Set[Tuple[int, int]] = set()
    for token_items in inverted.values():
        if len(token_items) <= 1:
            continue
        for i in range(len(token_items)):
            left = token_items[i]
            for j in range(i + 1, len(token_items)):
                right = token_items[j]
                if left < right:
                    pairs.add((left, right))
                else:
                    pairs.add((right, left))
    return pairs


def _prune_frequent_tokens(items: List[Set[str]], max_doc_frequency_ratio: float = 0.03) -> List[Set[str]]:
    if not items:
        return items
    n = len(items)
    if n < 5:
        return items

    df: Dict[str, int] = defaultdict(int)
    for toks in items:
        for tok in toks:
            df[tok] += 1

    max_df = max(2, int(n * max_doc_frequency_ratio))
    return [{tok for tok in toks if df.get(tok, 0) <= max_df} for toks in items]


def _cluster_by_similarity(items: List[Set[str]], threshold: float) -> List[int]:
    n = len(items)
    uf = _UnionFind(n)

    for i, j in _candidate_pairs(items):
        left, right = items[i], items[j]
        if not left or not right:
            continue
        if len(left) < 6 or len(right) < 6:
            continue
        sim = len(left & right) / max(1, len(left | right))
        if sim >= threshold:
            uf.union(i, j)

    root_to_cluster: Dict[int, int] = {}
    cluster_ids: List[int] = []
    next_id = 1
    for i in range(n):
        root = uf.find(i)
        if root not in root_to_cluster:
            root_to_cluster[root] = next_id
            next_id += 1
        cluster_ids.append(root_to_cluster[root])
    return cluster_ids


@dataclass
class DatasetContext:
    text_clusters: Dict[str, Any]
    image_clusters: Dict[str, Any]


def build_dataset_context(
    questions: List[Dict[str, Any]],
    *,
    image_store: Optional[Any],
    knowledge_base: Optional[Any],
    text_similarity_threshold: float,
) -> DatasetContext:
    text_sets: List[Set[str]] = []
    for q in questions:
        parts = [q.get("questionText", "")]
        for a in q.get("answers") or []:
            parts.append(a.get("text", ""))
        parts.append(q.get("explanationText", ""))
        text_sets.append(_tokenize("\n".join(parts)))

    text_sets = _prune_frequent_tokens(text_sets)
    text_cluster_ids = _cluster_by_similarity(text_sets, text_similarity_threshold)
    question_text_cluster: Dict[str, int] = {}
    cluster_to_question_ids: Dict[int, List[str]] = defaultdict(list)
    for q, cid in zip(questions, text_cluster_ids):
        qid = str(q.get("id") or "")
        question_text_cluster[qid] = cid
        cluster_to_question_ids[cid].append(qid)

    image_cluster_payload: Dict[str, Any] = {
        "enabled": bool(image_store),
        "questionImageClusters": {},
        "knowledgeImageMatches": {},
    }
    if image_store is not None:
        clusters = image_store.build_image_clusters(questions)
        image_cluster_payload["questionImageClusters"] = clusters
        if knowledge_base is not None:
            image_cluster_payload["knowledgeImageMatches"] = image_store.match_knowledge_images(
                questions,
                knowledge_base,
            )

    return DatasetContext(
        text_clusters={
            "questionToCluster": question_text_cluster,
            "clusterMembers": {str(k): v for k, v in cluster_to_question_ids.items()},
        },
        image_clusters=image_cluster_payload,
    )


def cluster_abstractions(
    questions: List[Dict[str, Any]],
    *,
    threshold: float,
) -> Dict[str, Any]:
    abstractions: List[str] = []
    question_ids: List[str] = []
    for q in questions:
        qid = str(q.get("id") or "")
        abstraction = (((q.get("aiAudit") or {}).get("questionAbstraction") or {}).get("summary") or "").strip()
        if not abstraction:
            abstraction = (q.get("questionText") or "").strip()
        abstractions.append(abstraction)
        question_ids.append(qid)

    abstraction_sets = [_tokenize(x) for x in abstractions]
    cluster_ids = _cluster_by_similarity(abstraction_sets, threshold)
    q_to_cluster = {qid: cid for qid, cid in zip(question_ids, cluster_ids)}
    cluster_to_qids: Dict[int, List[str]] = defaultdict(list)
    for qid, cid in q_to_cluster.items():
        cluster_to_qids[cid].append(qid)

    return {
        "questionToAbstractionCluster": q_to_cluster,
        "abstractionClusterMembers": {str(k): v for k, v in cluster_to_qids.items()},
    }
