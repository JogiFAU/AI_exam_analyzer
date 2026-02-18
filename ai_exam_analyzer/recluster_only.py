"""CLI to rerun only clustering on an existing dataset."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

from ai_exam_analyzer.config import CONFIG
from ai_exam_analyzer.image_store import QuestionImageStore
from ai_exam_analyzer.io_utils import load_json, save_json
from ai_exam_analyzer.workflow_context import build_dataset_context, cluster_abstractions


def _derive_output_path(input_path: str, requested_output: str) -> str:
    requested_output = (requested_output or "").strip()
    if requested_output:
        return requested_output

    input_dir = os.path.dirname(input_path)
    input_name = os.path.basename(input_path)
    stem, ext = os.path.splitext(input_name)
    ext = ext or ".json"
    output_name = f"{stem}.reclustered{ext}"
    return os.path.join(input_dir, output_name) if input_dir else output_name


def _extract_questions(data: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    if isinstance(data, dict) and "questions" in data:
        return data["questions"], data
    if isinstance(data, list):
        return data, None
    raise ValueError("Input must be a list of questions or {questions:[...]} object.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Rerun only clustering on an existing dataset (without Pass A/B/Review).",
    )
    ap.add_argument("--input", required=True, help="Input JSON (typically existing AIannotated file)")
    ap.add_argument("--output", default="", help="Output JSON path (default: <input>.reclustered.json)")

    ap.add_argument(
        "--text-cluster-similarity",
        type=float,
        default=0.12,
        help=(
            "Weighted-Jaccard threshold for question-content clustering. "
            f"Default 0.12 (relaxed vs pipeline default {CONFIG['TEXT_CLUSTER_SIMILARITY']})."
        ),
    )
    ap.add_argument(
        "--abstraction-cluster-similarity",
        type=float,
        default=0.18,
        help=(
            "Weighted-Jaccard threshold for abstraction clustering. "
            f"Default 0.18 (relaxed vs pipeline default {CONFIG['ABSTRACTION_CLUSTER_SIMILARITY']})."
        ),
    )
    ap.add_argument(
        "--images-zip",
        default="",
        help="Optional images ZIP to refresh questionImageClusterIds. If empty, image clustering is skipped.",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.output = _derive_output_path(args.input, args.output)

    data = load_json(args.input)
    questions, container = _extract_questions(data)

    image_store = None
    if args.images_zip:
        if not os.path.exists(args.images_zip):
            raise FileNotFoundError(f"--images-zip file not found: {args.images_zip}")
        image_store = QuestionImageStore.from_zip(args.images_zip)

    dataset_context = build_dataset_context(
        questions,
        image_store=image_store,
        knowledge_base=None,
        text_similarity_threshold=float(args.text_cluster_similarity),
    )
    abstraction_clusters = cluster_abstractions(
        questions,
        threshold=float(args.abstraction_cluster_similarity),
    )

    question_to_content = dataset_context.text_clusters.get("questionToCluster") or {}
    question_to_abstraction = abstraction_clusters.get("questionToAbstractionCluster") or {}
    question_to_image = (
        ((dataset_context.image_clusters.get("questionImageClusters") or {}).get("questionToClusters") or {})
        if image_store is not None
        else {}
    )

    total_questions = len(questions)
    for idx, q in enumerate(questions, start=1):
        qid = str(q.get("id") or "")
        print(f"[{idx}/{total_questions}] Reclustering gestartet für Frage {qid}.")
        audit = q.setdefault("aiAudit", {})
        clusters = audit.setdefault("clusters", {})
        clusters["questionContentClusterId"] = question_to_content.get(qid)
        clusters["abstractionClusterId"] = question_to_abstraction.get(qid)
        if image_store is not None:
            clusters["questionImageClusterIds"] = question_to_image.get(qid, [])
        print(f"[{idx}/{total_questions}] Reclustering abgeschlossen für Frage {qid}.")

    if isinstance(container, dict):
        container_meta = container.setdefault("meta", {})
        container_meta["clusteringRerun"] = {
            "textClusterSimilarity": float(args.text_cluster_similarity),
            "abstractionClusterSimilarity": float(args.abstraction_cluster_similarity),
            "imagesZipUsed": bool(image_store),
        }
        save_json(args.output, container)
    else:
        save_json(args.output, questions)

    print(
        f"Reclustering completed for {len(questions)} questions -> {args.output} "
        f"(text={args.text_cluster_similarity}, abstraction={args.abstraction_cluster_similarity}, "
        f"images={bool(image_store)})"
    )


if __name__ == "__main__":
    main()
