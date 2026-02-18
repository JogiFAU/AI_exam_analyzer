"""Command line interface for the analyzer pipeline."""

import argparse
import os

from ai_exam_analyzer.config import CONFIG
from ai_exam_analyzer.image_store import QuestionImageStore
from ai_exam_analyzer.io_utils import load_json
from ai_exam_analyzer.knowledge_base import (
    build_knowledge_base_from_zip,
    load_index_json,
    save_index_json,
)
from ai_exam_analyzer.processor import process_questions
from ai_exam_analyzer.schemas import schema_pass_a, schema_pass_b, schema_review_pass
from ai_exam_analyzer.topic_catalog import build_topic_catalog, format_topic_catalog_for_prompt


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", default=CONFIG["INPUT_PATH"], help="Input dataset JSON (original or in-progress annotated)")
    ap.add_argument("--topics", default=CONFIG["TOPICS_PATH"], help="Topic tree JSON (superTopics/subtopics)")
    ap.add_argument("--output", default=CONFIG["OUTPUT_PATH"], help="Output JSON path")

    ap.add_argument("--resume", dest="resume", action="store_true", default=CONFIG["RESUME"],
                    help="Skip questions already completed with this pipelineVersion")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="Do not skip completed questions")
    ap.add_argument("--limit", type=int, default=CONFIG["LIMIT"], help="Only process first N questions (0 = all)")
    ap.add_argument("--checkpoint-every", type=int, default=CONFIG["CHECKPOINT_EVERY"],
                    help="Save after every N processed questions")
    ap.add_argument("--sleep", type=float, default=CONFIG["SLEEP"], help="Sleep seconds between questions")

    ap.add_argument("--passA-model", default=CONFIG["PASSA_MODEL"])
    ap.add_argument("--passB-model", default=CONFIG["PASSB_MODEL"])
    ap.add_argument("--passA-temperature", type=float, default=CONFIG["PASSA_TEMPERATURE"])
    ap.add_argument("--passB-reasoning-effort", default=CONFIG["PASSB_REASONING_EFFORT"],
                    choices=["low", "medium", "high"])

    ap.add_argument("--trigger-answer-conf", type=float, default=CONFIG["TRIGGER_ANSWER_CONF"])
    ap.add_argument("--trigger-topic-conf", type=float, default=CONFIG["TRIGGER_TOPIC_CONF"])

    ap.add_argument("--apply-change-min-conf-b", type=float, default=CONFIG["APPLY_CHANGE_MIN_CONF_B"])
    ap.add_argument("--low-conf-maintenance-threshold", type=float,
                    default=CONFIG["LOW_CONF_MAINTENANCE_THRESHOLD"],
                    help="Below this confidence, auto-flag question as maintenance candidate")

    ap.add_argument("--write-top-level", dest="write_top_level", action="store_true",
                    default=CONFIG["WRITE_TOP_LEVEL"],
                    help="Also write ai* convenience fields on question level")
    ap.add_argument("--no-write-top-level", dest="write_top_level", action="store_false",
                    help="Do not write ai* convenience fields")

    ap.add_argument("--cleanup-spec", default=CONFIG["CLEANUP_SPEC_PATH"],
                    help="Optional JSON whitelist spec to keep only selected fields in output")

    ap.add_argument("--images-zip", default=CONFIG["IMAGES_ZIP_PATH"],
                    help="Optional ZIP with question images (default: images.zip)")

    ap.add_argument("--knowledge-zip", default=CONFIG["KNOWLEDGE_ZIP_PATH"],
                    help="Optional ZIP with PDFs/TXT/MD files used as retrieval knowledge base")
    ap.add_argument("--knowledge-index", default=CONFIG["KNOWLEDGE_INDEX_PATH"],
                    help="Optional path to load/save parsed knowledge index JSON")
    ap.add_argument("--knowledge-subject-hint", default=CONFIG["KNOWLEDGE_SUBJECT_HINT"],
                    help="Optional subject hint for knowledge-base matching")
    ap.add_argument("--knowledge-top-k", type=int, default=CONFIG["KNOWLEDGE_TOP_K"],
                    help="How many evidence chunks to include per question")
    ap.add_argument("--knowledge-max-chars", type=int, default=CONFIG["KNOWLEDGE_MAX_CHARS"],
                    help="Max cumulative evidence characters sent per question")
    ap.add_argument("--knowledge-min-score", type=float, default=CONFIG["KNOWLEDGE_MIN_SCORE"],
                    help="Minimum retrieval score for evidence chunk inclusion")
    ap.add_argument("--knowledge-chunk-chars", type=int, default=CONFIG["KNOWLEDGE_CHUNK_CHARS"],
                    help="Chunk size when parsing files from knowledge ZIP")

    ap.add_argument("--text-cluster-similarity", type=float, default=CONFIG["TEXT_CLUSTER_SIMILARITY"],
                    help="Weighted-Jaccard threshold for question-content clustering (with retrieval top-k candidates)")
    ap.add_argument("--abstraction-cluster-similarity", type=float, default=CONFIG["ABSTRACTION_CLUSTER_SIMILARITY"],
                    help="Weighted-Jaccard threshold for abstraction clustering")

    ap.add_argument("--enable-review-pass", dest="enable_review_pass", action="store_true",
                    default=CONFIG["ENABLE_REVIEW_PASS"],
                    help="Enable optional deep review pass for maintenance-heavy questions")
    ap.add_argument("--no-enable-review-pass", dest="enable_review_pass", action="store_false",
                    help="Disable optional deep review pass")
    ap.add_argument("--review-model", default=CONFIG["REVIEW_MODEL"])
    ap.add_argument("--review-min-maintenance-severity", type=int, default=CONFIG["REVIEW_MIN_MAINTENANCE_SEVERITY"],
                    help="Run review pass only for maintenance severity >= this value")
    ap.add_argument("--topic-candidate-top-k", type=int, default=CONFIG["TOPIC_CANDIDATE_TOP_K"],
                    help="How many deterministic topic candidates to attach per question")
    ap.add_argument("--run-report", default=CONFIG["RUN_REPORT_PATH"],
                    help="Optional JSON path for workflow run metrics/report")
    ap.add_argument("--topic-candidate-outside-force-passb-conf", type=float,
                    default=CONFIG["TOPIC_CANDIDATE_OUTSIDE_FORCE_PASSB_CONF"],
                    help="Run Pass B when Pass A picks topic outside candidates below this confidence")
    ap.add_argument("--enable-repeat-reconstruction", dest="enable_repeat_reconstruction", action="store_true",
                    default=CONFIG["ENABLE_REPEAT_RECONSTRUCTION"],
                    help="Enable repeat-pattern reconstruction suggestions across exam years")
    ap.add_argument("--no-enable-repeat-reconstruction", dest="enable_repeat_reconstruction", action="store_false",
                    help="Disable repeat-pattern reconstruction suggestions")
    ap.add_argument("--auto-apply-repeat-reconstruction", dest="auto_apply_repeat_reconstruction", action="store_true",
                    default=CONFIG["AUTO_APPLY_REPEAT_RECONSTRUCTION"],
                    help="Automatically apply repeat reconstruction suggestions when allowed by preprocessing gates")
    ap.add_argument("--no-auto-apply-repeat-reconstruction", dest="auto_apply_repeat_reconstruction", action="store_false",
                    help="Do not auto-apply repeat reconstruction suggestions")
    ap.add_argument("--repeat-min-similarity", type=float, default=CONFIG["REPEAT_MIN_SIMILARITY"],
                    help="Minimum similarity for repeat-pattern clustering")
    ap.add_argument("--repeat-min-anchor-conf", type=float, default=CONFIG["REPEAT_MIN_ANCHOR_CONF"],
                    help="Minimum combined confidence for high-quality repeat anchors")

    ap.add_argument("--debug", dest="debug", action="store_true", default=CONFIG["DEBUG"],
                    help="Store raw pass outputs under aiAudit._debug")
    ap.add_argument("--no-debug", dest="debug", action="store_false",
                    help="Do not store raw pass outputs")
    return ap


def _derive_output_path(input_path: str, requested_output: str) -> str:
    requested_output = (requested_output or "").strip()
    if requested_output:
        return requested_output

    input_path = (input_path or "").strip()
    input_dir = os.path.dirname(input_path)
    input_name = os.path.basename(input_path)
    stem, _ = os.path.splitext(input_name)
    stem = stem or "export"
    output_name = f"{stem} AIannotated.json"
    return os.path.join(input_dir, output_name) if input_dir else output_name


def main() -> None:
    args = build_parser().parse_args()
    args.output = _derive_output_path(args.input, args.output)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    topic_tree = load_json(args.topics)
    catalog, key_map = build_topic_catalog(topic_tree)
    topic_keys = [row["topicKey"] for row in catalog]
    topic_catalog_text = format_topic_catalog_for_prompt(catalog)

    schema_a = schema_pass_a(topic_keys)
    schema_b = schema_pass_b(topic_keys)
    schema_review = schema_review_pass(topic_keys)

    data = load_json(args.input)

    if isinstance(data, dict) and "questions" in data:
        questions = data["questions"]
        container = data
    elif isinstance(data, list):
        questions = data
        container = None
    else:
        raise ValueError("Input must be a list of questions or {questions:[...]} object.")

    cleanup_spec = None
    if args.cleanup_spec:
        cleanup_spec = load_json(args.cleanup_spec)

    subject_hint = args.knowledge_subject_hint
    if not subject_hint and isinstance(topic_tree, dict):
        super_topics = topic_tree.get("superTopics") or []
        if isinstance(super_topics, list) and super_topics:
            subject_hint = (super_topics[0].get("name") or "").strip()

    image_store = None
    if args.images_zip:
        if os.path.exists(args.images_zip):
            image_store = QuestionImageStore.from_zip(args.images_zip)
        elif args.images_zip != CONFIG["IMAGES_ZIP_PATH"]:
            raise FileNotFoundError(f"--images-zip file not found: {args.images_zip}")

    knowledge_base = None
    if args.knowledge_index:
        index_path = args.knowledge_index
        if os.path.exists(index_path):
            knowledge_base = load_index_json(index_path)
        elif args.knowledge_zip:
            knowledge_base = build_knowledge_base_from_zip(
                args.knowledge_zip,
                max_chunk_chars=args.knowledge_chunk_chars,
                subject_hint=subject_hint,
            )
            save_index_json(index_path, knowledge_base)
        else:
            raise ValueError("--knowledge-index was provided but file does not exist and --knowledge-zip is missing.")
    elif args.knowledge_zip:
        knowledge_base = build_knowledge_base_from_zip(
            args.knowledge_zip,
            max_chunk_chars=args.knowledge_chunk_chars,
            subject_hint=subject_hint,
        )

    process_questions(
        args=args,
        questions=questions,
        container=container,
        key_map=key_map,
        topic_catalog_text=topic_catalog_text,
        topic_catalog=catalog,
        schema_a=schema_a,
        schema_b=schema_b,
        schema_review=schema_review,
        cleanup_spec=cleanup_spec,
        knowledge_base=knowledge_base,
        image_store=image_store,
    )


if __name__ == "__main__":
    main()
