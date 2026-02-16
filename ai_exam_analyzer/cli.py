"""Command line interface for the analyzer pipeline."""

import argparse
import os

from ai_exam_analyzer.config import CONFIG
from ai_exam_analyzer.io_utils import load_json
from ai_exam_analyzer.processor import process_questions
from ai_exam_analyzer.schemas import schema_pass_a, schema_pass_b
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

    ap.add_argument("--debug", dest="debug", action="store_true", default=CONFIG["DEBUG"],
                    help="Store raw pass outputs under aiAudit._debug")
    ap.add_argument("--no-debug", dest="debug", action="store_false",
                    help="Do not store raw pass outputs")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    topic_tree = load_json(args.topics)
    catalog, key_map = build_topic_catalog(topic_tree)
    topic_keys = [row["topicKey"] for row in catalog]
    topic_catalog_text = format_topic_catalog_for_prompt(catalog)

    schema_a = schema_pass_a(topic_keys)
    schema_b = schema_pass_b(topic_keys)

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

    process_questions(
        args=args,
        questions=questions,
        container=container,
        key_map=key_map,
        topic_catalog_text=topic_catalog_text,
        schema_a=schema_a,
        schema_b=schema_b,
        cleanup_spec=cleanup_spec,
    )


if __name__ == "__main__":
    main()
