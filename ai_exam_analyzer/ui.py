"""Streamlit UI for local execution of the AI exam analyzer."""

import os
from types import SimpleNamespace
from typing import Any, Dict, Optional

import streamlit as st

from ai_exam_analyzer.config import CONFIG
from ai_exam_analyzer.io_utils import load_json
from ai_exam_analyzer.knowledge_base import (
    build_knowledge_base_from_zip,
    load_index_json,
    save_index_json,
)
from ai_exam_analyzer.processor import process_questions
from ai_exam_analyzer.schemas import schema_pass_a, schema_pass_b
from ai_exam_analyzer.topic_catalog import build_topic_catalog, format_topic_catalog_for_prompt


def _resolve_path(*, folder: str, filename: str) -> str:
    folder = (folder or "").strip()
    filename = (filename or "").strip()

    if not filename:
        return ""
    if os.path.isabs(filename):
        return filename
    if folder:
        return os.path.join(folder, filename)
    return filename


def _build_args() -> SimpleNamespace:
    default_data_dir = os.path.expanduser("~")

    input_default_name = os.path.basename(CONFIG["INPUT_PATH"]) or "export.json"
    topics_default_name = os.path.basename(CONFIG["TOPICS_PATH"]) or "topic-tree.json"
    output_default_name = os.path.basename(CONFIG["OUTPUT_PATH"]) or "export.AIannotated.json"
    cleanup_default_name = os.path.basename(CONFIG["CLEANUP_SPEC_PATH"]) or "whitelist.json"
    knowledge_zip_default_name = os.path.basename(CONFIG["KNOWLEDGE_ZIP_PATH"]) or "knowledge.zip"
    knowledge_index_default_name = os.path.basename(CONFIG["KNOWLEDGE_INDEX_PATH"]) or "knowledge.index.json"

    with st.sidebar:
        st.header("Einstellungen")
        st.subheader("Dateien & Ordner")
        data_folder = st.text_input(
            "Datenordner (Input)",
            value=default_data_dir,
            help="Ordner mit Input- und Konfigurationsdateien (z. B. export.json, topic-tree.json, whitelist.json, knowledge.zip).",
        ).strip()
        output_folder = st.text_input(
            "Ausgabeordner",
            value=data_folder or default_data_dir,
            help="Ordner, in dem die Output-Datei gespeichert wird.",
        ).strip()

        input_name = st.text_input("Input JSON Dateiname", value=input_default_name)
        topics_name = st.text_input("Topic-Tree JSON Dateiname", value=topics_default_name)
        output_name = st.text_input("Output JSON Dateiname", value=output_default_name)

        st.subheader("API")
        api_key_value = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API Key", type="password", value=api_key_value)

        st.subheader("Pipeline")
        resume = st.checkbox("Resume aktiv", value=CONFIG["RESUME"])
        limit = st.number_input("Limit (0 = alle Fragen)", min_value=0, value=int(CONFIG["LIMIT"]))
        checkpoint_every = st.number_input(
            "Checkpoint alle N Fragen",
            min_value=1,
            value=int(CONFIG["CHECKPOINT_EVERY"]),
        )
        sleep_seconds = st.number_input("Pause je Frage (Sek.)", min_value=0.0, value=float(CONFIG["SLEEP"]), step=0.05)

        pass_a_model = st.text_input("Pass A Modell", value=CONFIG["PASSA_MODEL"])
        pass_b_model = st.text_input("Pass B Modell", value=CONFIG["PASSB_MODEL"])
        pass_a_temperature = st.number_input("Pass A Temperature", min_value=0.0, max_value=2.0, value=float(CONFIG["PASSA_TEMPERATURE"]), step=0.1)
        pass_b_reasoning_effort = st.selectbox(
            "Pass B Reasoning Effort",
            options=["low", "medium", "high"],
            index=["low", "medium", "high"].index(CONFIG["PASSB_REASONING_EFFORT"]),
        )

        trigger_answer_conf = st.slider("Pass B Trigger: Answer Confidence", 0.0, 1.0, float(CONFIG["TRIGGER_ANSWER_CONF"]), 0.01)
        trigger_topic_conf = st.slider("Pass B Trigger: Topic Confidence", 0.0, 1.0, float(CONFIG["TRIGGER_TOPIC_CONF"]), 0.01)
        apply_change_min_conf_b = st.slider("Änderung anwenden ab Pass-B Confidence", 0.0, 1.0, float(CONFIG["APPLY_CHANGE_MIN_CONF_B"]), 0.01)
        low_conf_maintenance_threshold = st.slider(
            "Wartung markieren unter Confidence",
            0.0,
            1.0,
            float(CONFIG["LOW_CONF_MAINTENANCE_THRESHOLD"]),
            0.01,
        )

        write_top_level = st.checkbox("Top-Level ai* Felder schreiben", value=CONFIG["WRITE_TOP_LEVEL"])
        debug = st.checkbox("Debug-Rohdaten speichern", value=CONFIG["DEBUG"])
        use_cleanup_spec = st.checkbox("Whitelist/Cleanup anwenden", value=bool(CONFIG["CLEANUP_SPEC_PATH"]))
        cleanup_spec_name = st.text_input(
            "Whitelist/Cleanup JSON Dateiname (optional)",
            value=cleanup_default_name,
            disabled=not use_cleanup_spec,
        )

        st.subheader("Knowledge Base (optional)")
        use_knowledge_zip = st.checkbox("Knowledge ZIP verwenden", value=bool(CONFIG["KNOWLEDGE_ZIP_PATH"]))
        knowledge_zip_name = st.text_input(
            "Knowledge ZIP Dateiname (optional)",
            value=knowledge_zip_default_name,
            disabled=not use_knowledge_zip,
        )
        use_knowledge_index = st.checkbox("Knowledge Index verwenden", value=bool(CONFIG["KNOWLEDGE_INDEX_PATH"]))
        knowledge_index_name = st.text_input(
            "Knowledge Index JSON Dateiname (optional)",
            value=knowledge_index_default_name,
            disabled=not use_knowledge_index,
        )
        knowledge_subject_hint = st.text_input("Subject Hint", value=CONFIG["KNOWLEDGE_SUBJECT_HINT"])
        knowledge_top_k = st.number_input("Knowledge Top-K", min_value=1, value=int(CONFIG["KNOWLEDGE_TOP_K"]))
        knowledge_max_chars = st.number_input("Knowledge Max Chars", min_value=500, value=int(CONFIG["KNOWLEDGE_MAX_CHARS"]), step=100)
        knowledge_min_score = st.slider("Knowledge Min Score", 0.0, 1.0, float(CONFIG["KNOWLEDGE_MIN_SCORE"]), 0.01)
        knowledge_chunk_chars = st.number_input(
            "Knowledge Chunk Chars",
            min_value=200,
            value=int(CONFIG["KNOWLEDGE_CHUNK_CHARS"]),
            step=100,
        )

    input_path = _resolve_path(folder=data_folder, filename=input_name)
    topics_path = _resolve_path(folder=data_folder, filename=topics_name)
    output_path = _resolve_path(folder=output_folder, filename=output_name)
    cleanup_spec = _resolve_path(folder=data_folder, filename=cleanup_spec_name) if use_cleanup_spec else ""
    knowledge_zip = _resolve_path(folder=data_folder, filename=knowledge_zip_name) if use_knowledge_zip else ""
    knowledge_index = _resolve_path(folder=data_folder, filename=knowledge_index_name) if use_knowledge_index else ""

    return SimpleNamespace(
        input=input_path,
        topics=topics_path,
        output=output_path,
        api_key=api_key,
        resume=resume,
        limit=int(limit),
        checkpoint_every=int(checkpoint_every),
        sleep=float(sleep_seconds),
        passA_model=pass_a_model,
        passB_model=pass_b_model,
        passA_temperature=float(pass_a_temperature),
        passB_reasoning_effort=pass_b_reasoning_effort,
        trigger_answer_conf=float(trigger_answer_conf),
        trigger_topic_conf=float(trigger_topic_conf),
        apply_change_min_conf_b=float(apply_change_min_conf_b),
        low_conf_maintenance_threshold=float(low_conf_maintenance_threshold),
        write_top_level=write_top_level,
        debug=debug,
        cleanup_spec=cleanup_spec.strip(),
        knowledge_zip=knowledge_zip.strip(),
        knowledge_index=knowledge_index.strip(),
        knowledge_subject_hint=knowledge_subject_hint.strip(),
        knowledge_top_k=int(knowledge_top_k),
        knowledge_max_chars=int(knowledge_max_chars),
        knowledge_min_score=float(knowledge_min_score),
        knowledge_chunk_chars=int(knowledge_chunk_chars),
    )


def _prepare_knowledge_base(args: SimpleNamespace, topic_tree: Any) -> Optional[Any]:
    subject_hint = args.knowledge_subject_hint
    if not subject_hint and isinstance(topic_tree, dict):
        super_topics = topic_tree.get("superTopics") or []
        if isinstance(super_topics, list) and super_topics:
            subject_hint = (super_topics[0].get("name") or "").strip()

    knowledge_base = None
    if args.knowledge_index:
        if os.path.exists(args.knowledge_index):
            knowledge_base = load_index_json(args.knowledge_index)
        elif args.knowledge_zip:
            knowledge_base = build_knowledge_base_from_zip(
                args.knowledge_zip,
                max_chunk_chars=args.knowledge_chunk_chars,
                subject_hint=subject_hint,
            )
            save_index_json(args.knowledge_index, knowledge_base)
        else:
            raise ValueError("Knowledge-Index angegeben, aber Datei fehlt und Knowledge-ZIP wurde nicht gesetzt.")
    elif args.knowledge_zip:
        knowledge_base = build_knowledge_base_from_zip(
            args.knowledge_zip,
            max_chunk_chars=args.knowledge_chunk_chars,
            subject_hint=subject_hint,
        )

    return knowledge_base


def main() -> None:
    st.set_page_config(page_title="AI Exam Analyzer", layout="wide")
    st.title("AI Exam Analyzer – Lokale Oberfläche")
    st.caption("Konfiguration, API-Key und Live-Analysefortschritt in einer Oberfläche.")

    args = _build_args()

    start_button = st.button("Analyse starten", type="primary", use_container_width=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics = st.empty()
    event_log = st.empty()

    if not start_button:
        st.info("Setze deine Einstellungen und klicke auf **Analyse starten**.")
        return

    if not args.api_key:
        st.error("Bitte gib einen OpenAI API Key ein.")
        return

    os.environ["OPENAI_API_KEY"] = args.api_key

    try:
        topic_tree = load_json(args.topics)
        catalog, key_map = build_topic_catalog(topic_tree)
        topic_keys = [row["topicKey"] for row in catalog]
        topic_catalog_text = format_topic_catalog_for_prompt(catalog)

        schema_a = schema_pass_a(topic_keys)
        schema_b = schema_pass_b(topic_keys)

        data = load_json(args.input)
        if isinstance(data, dict) and "questions" in data:
            questions = data["questions"]
            container: Optional[Dict[str, Any]] = data
        elif isinstance(data, list):
            questions = data
            container = None
        else:
            raise ValueError("Input muss Liste oder Objekt mit 'questions' sein.")

        cleanup_spec = load_json(args.cleanup_spec) if args.cleanup_spec else None
        knowledge_base = _prepare_knowledge_base(args, topic_tree)

        recent_events = []

        def on_progress(event: Dict[str, Any]) -> None:
            total = max(1, int(event.get("total") or len(questions) or 1))
            processed = int(event.get("processed", 0))
            pct = min(1.0, processed / total)
            progress_bar.progress(pct)
            status_text.markdown(f"**Status:** {event.get('message', '---')}")
            metrics.metric("Fortschritt", f"{processed}/{total}")

            line = (
                f"- [{event.get('event', 'event')}] "
                f"{event.get('message', '')} "
                f"(done={event.get('done', 0)}, skipped={event.get('skipped', 0)})"
            )
            recent_events.append(line)
            if len(recent_events) > 12:
                del recent_events[0]
            event_log.markdown("**Live-Log**\n" + "\n".join(recent_events))

        process_questions(
            args=args,
            questions=questions,
            container=container,
            key_map=key_map,
            topic_catalog_text=topic_catalog_text,
            schema_a=schema_a,
            schema_b=schema_b,
            cleanup_spec=cleanup_spec,
            knowledge_base=knowledge_base,
            progress_callback=on_progress,
        )

        progress_bar.progress(1.0)
        st.success(f"Analyse beendet. Ergebnis gespeichert unter: {args.output}")

    except Exception as exc:
        st.exception(exc)


if __name__ == "__main__":
    main()
