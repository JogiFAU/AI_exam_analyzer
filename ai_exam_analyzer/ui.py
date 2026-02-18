"""Streamlit UI for local execution of the AI exam analyzer."""

import os
from types import SimpleNamespace
from typing import Any, Dict, Optional

import streamlit as st

from ai_exam_analyzer.config import CONFIG
from ai_exam_analyzer.image_store import QuestionImageStore
from ai_exam_analyzer.io_utils import load_json
from ai_exam_analyzer.knowledge_base import (
    build_knowledge_base_from_zip,
    load_index_json,
    save_index_json,
)
from ai_exam_analyzer.processor import process_questions
from ai_exam_analyzer.schemas import (
    schema_explainer_pass,
    schema_pass_a,
    schema_pass_b,
    schema_reconstruction_pass,
    schema_review_pass,
)
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


def _derive_output_path_from_input(input_path: str, output_folder: str = "") -> str:
    input_path = (input_path or "").strip()
    output_folder = (output_folder or "").strip()
    input_dir = os.path.dirname(input_path)
    input_name = os.path.basename(input_path)
    stem, _ = os.path.splitext(input_name)
    stem = stem or "export"
    output_name = f"{stem} AIannotated.json"

    if output_folder:
        return os.path.join(output_folder, output_name)
    return os.path.join(input_dir, output_name) if input_dir else output_name


def _get_default_documents_dir() -> str:
    home_dir = os.path.expanduser("~")
    documents_dir = os.path.join(home_dir, "Documents")
    if os.path.isdir(documents_dir):
        return documents_dir
    return home_dir


def _pick_directory(initial_dir: str) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(initialdir=initial_dir or os.path.expanduser("~"))
    root.destroy()
    return selected or None


def _pick_file(initial_dir: str) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(initialdir=initial_dir or os.path.expanduser("~"))
    root.destroy()
    return selected or None


def _infer_subject_hint_from_topic_tree(topics_path: str) -> str:
    topics_path = (topics_path or "").strip()
    if not topics_path or (not os.path.exists(topics_path)):
        return ""

    try:
        topic_tree = load_json(topics_path)
    except Exception:
        return ""

    if not isinstance(topic_tree, dict):
        return ""

    subject = topic_tree.get("subject")
    if isinstance(subject, str) and subject.strip():
        return subject.strip()
    return ""


def _file_picker_row(*, state_key: str, label: str, default_path: str, start_dir: str, help_text: str, optional: bool = False, require_existing: bool = True) -> str:
    widget_key = f"{state_key}_input"
    last_default_key = f"{state_key}_last_default"
    default_candidate = default_path if (default_path and (os.path.exists(default_path) or (not require_existing))) else ""

    if state_key not in st.session_state:
        st.session_state[state_key] = default_candidate
    if last_default_key not in st.session_state:
        st.session_state[last_default_key] = default_candidate

    previous_default = st.session_state.get(last_default_key, "")
    current_value = st.session_state.get(state_key, "")

    # Keep folder-derived defaults in sync while preserving manual overrides.
    if current_value == previous_default and current_value != default_candidate:
        st.session_state[state_key] = default_candidate
    elif not current_value and default_candidate:
        st.session_state[state_key] = default_candidate

    st.session_state[last_default_key] = default_candidate

    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state[state_key]
    if st.session_state.get(widget_key) != st.session_state.get(state_key):
        st.session_state[widget_key] = st.session_state[state_key]

    cols = st.columns([4, 1])
    with cols[0]:
        chosen = st.text_input(label, key=widget_key, help=help_text)
        st.session_state[state_key] = (chosen or "").strip()
    with cols[1]:
        if st.button("📂 Wählen", key=f"{state_key}_btn", help="Datei per Dialog auswählen"):
            picked = _pick_file(start_dir)
            if picked:
                st.session_state[state_key] = picked
                st.rerun()
            else:
                st.warning("Datei-Dialog konnte nicht geöffnet werden (z. B. kein GUI-Support).")

    if optional:
        return st.session_state.get(state_key, "").strip()

    if not st.session_state.get(state_key, "").strip():
        st.caption("❌ Noch keine Datei ausgewählt")
    return st.session_state.get(state_key, "").strip()


def _build_args() -> SimpleNamespace:
    if "data_folder" not in st.session_state:
        st.session_state["data_folder"] = _get_default_documents_dir()
    if "output_folder" not in st.session_state:
        st.session_state["output_folder"] = st.session_state["data_folder"]

    data_folder = st.session_state["data_folder"]
    output_folder = st.session_state["output_folder"]

    input_default_name = os.path.basename(CONFIG["INPUT_PATH"]) or "export.json"
    topics_default_name = os.path.basename(CONFIG["TOPICS_PATH"]) or "topic-tree.json"
    output_default_name = os.path.basename(CONFIG["OUTPUT_PATH"]) or ""
    cleanup_default_name = os.path.basename(CONFIG["CLEANUP_SPEC_PATH"]) or "whitelist.json"
    images_zip_default_name = os.path.basename(CONFIG["IMAGES_ZIP_PATH"]) or "images.zip"
    knowledge_zip_default_name = os.path.basename(CONFIG["KNOWLEDGE_ZIP_PATH"]) or "knowledge.zip"
    knowledge_index_default_name = os.path.basename(CONFIG["KNOWLEDGE_INDEX_PATH"]) or "knowledge.index.json"

    with st.sidebar:
        st.header("Einstellungen")

        with st.expander("📁 Datenquellen", expanded=True):
            st.caption("Datenordner (Standard für Dateiauswahl)")
            st.code(data_folder)
            if st.button("📁 Datenordner auswählen", key="pick_data_folder", help="Wählt den Hauptordner für Input-Dateien"):
                picked_dir = _pick_directory(data_folder)
                if picked_dir:
                    old_data_folder = st.session_state["data_folder"]
                    st.session_state["data_folder"] = picked_dir
                    if st.session_state.get("output_folder") == old_data_folder:
                        st.session_state["output_folder"] = picked_dir
                    st.rerun()
                else:
                    st.warning("Ordner-Dialog konnte nicht geöffnet werden (z. B. kein GUI-Support).")

            st.caption("Ausgabeordner")
            st.code(output_folder)
            if st.button("📁 Ausgabeordner auswählen", key="pick_output_folder", help="Wählt den Ordner für die Ausgabe-Datei"):
                picked_output_dir = _pick_directory(output_folder)
                if picked_output_dir:
                    st.session_state["output_folder"] = picked_output_dir
                    st.rerun()
                else:
                    st.warning("Ordner-Dialog konnte nicht geöffnet werden (z. B. kein GUI-Support).")

            output_status_name = output_default_name or os.path.basename(
                _derive_output_path_from_input(
                    _resolve_path(folder=data_folder, filename=input_default_name),
                    output_folder,
                )
            )
            defaults = [
                ("Input", input_default_name),
                ("Topic-Tree", topics_default_name),
                ("Output", output_status_name),
                ("Whitelist", cleanup_default_name),
                ("Bilder ZIP", images_zip_default_name),
                ("Knowledge ZIP", knowledge_zip_default_name),
            ]
            st.caption("Status im Datenordner (Standarddateien):")
            for label, name in defaults:
                path = _resolve_path(folder=data_folder, filename=name)
                icon = "✅" if os.path.exists(path) else "❌"
                st.caption(f"{icon} {label}: `{name}`")

            input_path = _file_picker_row(
                state_key="input_file",
                label="Input JSON",
                default_path=_resolve_path(folder=data_folder, filename=input_default_name),
                start_dir=data_folder,
                help_text="Datei mit Fragen (z. B. export.json).",
            )
            topics_path = _file_picker_row(
                state_key="topics_file",
                label="Topic-Tree JSON",
                default_path=_resolve_path(folder=data_folder, filename=topics_default_name),
                start_dir=data_folder,
                help_text="Topic-Struktur-Datei (z. B. topic-tree.json).",
            )
            derived_output_path = _derive_output_path_from_input(input_path, output_folder)
            if output_default_name:
                output_default_path = _resolve_path(folder=output_folder, filename=output_default_name)
            else:
                output_default_path = derived_output_path

            output_path = _file_picker_row(
                state_key="output_file",
                label="Output JSON",
                default_path=output_default_path,
                start_dir=output_folder,
                help_text="Zieldatei für annotierte Ausgabe (wird automatisch erstellt).",
                require_existing=False,
            )

            use_cleanup_spec = st.checkbox(
                "Whitelist/Cleanup nutzen",
                value=bool(CONFIG["CLEANUP_SPEC_PATH"]),
                help="Wenn aktiv, wird eine Whitelist/Cleanup-Spec auf das Ausgabeformat angewendet.",
            )
            cleanup_spec = _file_picker_row(
                state_key="cleanup_file",
                label="Whitelist/Cleanup JSON",
                default_path=_resolve_path(folder=data_folder, filename=cleanup_default_name),
                start_dir=data_folder,
                help_text="Optionale whitelist.json bzw. Cleanup-Spec.",
                optional=True,
            ) if use_cleanup_spec else ""

            use_images_zip = st.checkbox(
                "Fragenbilder ZIP nutzen",
                value=bool(CONFIG["IMAGES_ZIP_PATH"]),
                help="Wenn aktiv, werden Fragebilder aus einer ZIP geladen und dem Modell mitgegeben.",
            )
            images_zip = _file_picker_row(
                state_key="images_zip_file",
                label="Fragenbilder ZIP",
                default_path=_resolve_path(folder=data_folder, filename=images_zip_default_name),
                start_dir=data_folder,
                help_text="ZIP mit Fragebildern (Dateinamen enthalten die Frage-ID).",
                optional=True,
            ) if use_images_zip else ""

            use_knowledge_zip = st.checkbox(
                "Knowledge ZIP nutzen",
                value=bool(CONFIG["KNOWLEDGE_ZIP_PATH"]),
                help="Wenn aktiv, wird Wissen aus einer ZIP-Datei geladen.",
            )
            knowledge_zip = _file_picker_row(
                state_key="knowledge_zip_file",
                label="Knowledge ZIP",
                default_path=_resolve_path(folder=data_folder, filename=knowledge_zip_default_name),
                start_dir=data_folder,
                help_text="ZIP mit Wissensdokumenten (PDF/TXT/MD).",
                optional=True,
            ) if use_knowledge_zip else ""

            use_knowledge_index = st.checkbox(
                "Knowledge-Index nutzen",
                value=bool(CONFIG["KNOWLEDGE_INDEX_PATH"]),
                help="Optionaler Index für schnelleren Start; wird geladen/geschrieben.",
            )
            knowledge_index = _file_picker_row(
                state_key="knowledge_index_file",
                label="Knowledge Index JSON",
                default_path=_resolve_path(folder=data_folder, filename=knowledge_index_default_name),
                start_dir=data_folder,
                help_text="Optionaler Index-Cache als JSON.",
                optional=True,
            ) if use_knowledge_index else ""

            inferred_subject_hint = _infer_subject_hint_from_topic_tree(topics_path)
            subject_hint_default = inferred_subject_hint or CONFIG["KNOWLEDGE_SUBJECT_HINT"]
            if "knowledge_subject_hint" not in st.session_state:
                st.session_state["knowledge_subject_hint"] = subject_hint_default
            if "knowledge_subject_hint_last_default" not in st.session_state:
                st.session_state["knowledge_subject_hint_last_default"] = subject_hint_default

            previous_subject_default = st.session_state["knowledge_subject_hint_last_default"]
            current_subject_hint = st.session_state["knowledge_subject_hint"]
            if current_subject_hint == previous_subject_default and current_subject_hint != subject_hint_default:
                st.session_state["knowledge_subject_hint"] = subject_hint_default
            elif (not current_subject_hint) and subject_hint_default:
                st.session_state["knowledge_subject_hint"] = subject_hint_default
            st.session_state["knowledge_subject_hint_last_default"] = subject_hint_default

        with st.expander("🔐 API", expanded=True):
            api_key_value = os.getenv("OPENAI_API_KEY", "")
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=api_key_value,
                help="API-Key für den Zugriff auf die OpenAI-Modelle.",
            )

        with st.expander("⚙️ Pipeline", expanded=False):
            resume = st.checkbox("Resume aktiv", value=CONFIG["RESUME"], help="Überspringt bereits abgeschlossene Fragen.")
            limit = st.number_input("Limit (0 = alle Fragen)", min_value=0, value=int(CONFIG["LIMIT"]), help="Begrenzt die Anzahl verarbeiteter Fragen.")
            checkpoint_every = st.number_input(
                "Checkpoint alle N Fragen",
                min_value=1,
                value=int(CONFIG["CHECKPOINT_EVERY"]),
                help="Speichert regelmäßig Zwischenergebnisse.",
            )
            sleep_seconds = st.number_input(
                "Pause je Frage (Sek.)",
                min_value=0.0,
                value=float(CONFIG["SLEEP"]),
                step=0.05,
                help="Kurze Pause zwischen zwei API-Aufrufen.",
            )

            pass_a_model = st.text_input("Pass A Modell", value=CONFIG["PASSA_MODEL"], help="Modell für Erstbewertung.")
            pass_b_model = st.text_input("Pass B Modell", value=CONFIG["PASSB_MODEL"], help="Modell für Verifikation/Review.")
            pass_a_temperature = st.number_input(
                "Pass A Temperature",
                min_value=0.0,
                max_value=2.0,
                value=float(CONFIG["PASSA_TEMPERATURE"]),
                step=0.1,
                help="Sampling-Temperatur für Pass A.",
            )
            pass_b_reasoning_effort = st.selectbox(
                "Pass B Reasoning Effort",
                options=["low", "medium", "high"],
                index=["low", "medium", "high"].index(CONFIG["PASSB_REASONING_EFFORT"]),
                help="Rechenaufwand für Pass B.",
            )

            trigger_answer_conf = st.slider(
                "Pass B Trigger: Answer Confidence",
                0.0,
                1.0,
                float(CONFIG["TRIGGER_ANSWER_CONF"]),
                0.01,
                help="Unterhalb dieses Werts wird Pass B ausgelöst (Antwort-Vertrauen).",
            )
            trigger_topic_conf = st.slider(
                "Pass B Trigger: Topic Confidence",
                0.0,
                1.0,
                float(CONFIG["TRIGGER_TOPIC_CONF"]),
                0.01,
                help="Unterhalb dieses Werts wird Pass B ausgelöst (Topic-Vertrauen).",
            )
            apply_change_min_conf_b = st.slider(
                "Änderung anwenden ab Pass-B Confidence",
                0.0,
                1.0,
                float(CONFIG["APPLY_CHANGE_MIN_CONF_B"]),
                0.01,
                help="Mindestvertrauen von Pass B, damit Antwortänderungen übernommen werden.",
            )
            low_conf_maintenance_threshold = st.slider(
                "Wartung markieren unter Confidence",
                0.0,
                1.0,
                float(CONFIG["LOW_CONF_MAINTENANCE_THRESHOLD"]),
                0.01,
                help="Unterhalb dieses Werts wird die Frage als Wartungsfall markiert.",
            )

            text_cluster_similarity = st.slider(
                "Question-Cluster Similarity",
                0.0,
                1.0,
                float(CONFIG["TEXT_CLUSTER_SIMILARITY"]),
                0.01,
                help="Ähnlichkeitsschwelle für inhaltliche Frage-Cluster (Jaccard).",
            )
            abstraction_cluster_similarity = st.slider(
                "Abstraction-Cluster Similarity",
                0.0,
                1.0,
                float(CONFIG["ABSTRACTION_CLUSTER_SIMILARITY"]),
                0.01,
                help="Ähnlichkeitsschwelle für Cluster der Frageabstraktionen.",
            )

            enable_review_pass = st.checkbox(
                "Pass C (Deep Review) aktivieren",
                value=bool(CONFIG["ENABLE_REVIEW_PASS"]),
                help="Optionaler dritter Review-Pass für wartungsintensive Fragen.",
            )
            review_model = st.text_input(
                "Pass C Modell",
                value=CONFIG["REVIEW_MODEL"],
                help="Modell für den optionalen Deep-Review-Pass.",
                disabled=not enable_review_pass,
            )
            review_min_maintenance_severity = st.select_slider(
                "Pass C ab Wartungs-Severity",
                options=[1, 2, 3],
                value=int(CONFIG["REVIEW_MIN_MAINTENANCE_SEVERITY"]),
                help="Pass C läuft nur ab diesem Wartungs-Schweregrad.",
                disabled=not enable_review_pass,
            )

            write_top_level = st.checkbox(
                "Top-Level ai* Felder schreiben",
                value=CONFIG["WRITE_TOP_LEVEL"],
                help="Schreibt zusätzliche ai*-Felder direkt in jede Frage.",
            )
            debug = st.checkbox(
                "Debug-Rohdaten speichern",
                value=CONFIG["DEBUG"],
                help="Speichert detaillierte Rohantworten unter aiAudit._debug.",
            )

        with st.expander("🧠 Knowledge Base", expanded=False):
            knowledge_subject_hint = st.text_input(
                "Subject Hint",
                key="knowledge_subject_hint",
                help="Standard wird aus dem Topic-Tree übernommen; kann hier manuell überschrieben werden.",
            )
            knowledge_top_k = st.number_input(
                "Knowledge Top-K",
                min_value=1,
                value=int(CONFIG["KNOWLEDGE_TOP_K"]),
                help="Anzahl der Beleg-Chunks pro Frage.",
            )
            knowledge_max_chars = st.number_input(
                "Knowledge Max Chars",
                min_value=500,
                value=int(CONFIG["KNOWLEDGE_MAX_CHARS"]),
                step=100,
                help="Maximale Gesamtlänge der übergebenen Belege.",
            )
            knowledge_min_score = st.slider(
                "Knowledge Min Score",
                0.0,
                1.0,
                float(CONFIG["KNOWLEDGE_MIN_SCORE"]),
                0.01,
                help="Mindestrelevanz eines Chunks.",
            )
            knowledge_chunk_chars = st.number_input(
                "Knowledge Chunk Chars",
                min_value=200,
                value=int(CONFIG["KNOWLEDGE_CHUNK_CHARS"]),
                step=100,
                help="Chunk-Größe beim Parsen der ZIP-Datei.",
            )

    return SimpleNamespace(
        input=input_path,
        topics=topics_path,
        output=(output_path or _derive_output_path_from_input(input_path, output_folder)),
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
        images_zip=images_zip.strip(),
        knowledge_zip=knowledge_zip.strip(),
        knowledge_index=knowledge_index.strip(),
        knowledge_subject_hint=knowledge_subject_hint.strip(),
        knowledge_top_k=int(knowledge_top_k),
        knowledge_max_chars=int(knowledge_max_chars),
        knowledge_min_score=float(knowledge_min_score),
        knowledge_chunk_chars=int(knowledge_chunk_chars),
        text_cluster_similarity=float(text_cluster_similarity),
        abstraction_cluster_similarity=float(abstraction_cluster_similarity),
        enable_review_pass=bool(enable_review_pass),
        review_model=review_model.strip(),
        review_min_maintenance_severity=int(review_min_maintenance_severity),
        topic_candidate_top_k=int(CONFIG["TOPIC_CANDIDATE_TOP_K"]),
        run_report=str(CONFIG.get("RUN_REPORT_PATH", "")),
        topic_candidate_outside_force_passb_conf=float(CONFIG["TOPIC_CANDIDATE_OUTSIDE_FORCE_PASSB_CONF"]),
        enable_repeat_reconstruction=bool(CONFIG["ENABLE_REPEAT_RECONSTRUCTION"]),
        auto_apply_repeat_reconstruction=bool(CONFIG["AUTO_APPLY_REPEAT_RECONSTRUCTION"]),
        repeat_min_similarity=float(CONFIG["REPEAT_MIN_SIMILARITY"]),
        repeat_min_anchor_conf=float(CONFIG["REPEAT_MIN_ANCHOR_CONF"]),
        repeat_min_anchor_consensus=int(CONFIG["REPEAT_MIN_ANCHOR_CONSENSUS"]),
        repeat_min_match_ratio=float(CONFIG["REPEAT_MIN_MATCH_RATIO"]),
        enable_reconstruction_pass=bool(CONFIG["ENABLE_RECONSTRUCTION_PASS"]),
        reconstruction_model=str(CONFIG["RECONSTRUCTION_MODEL"]),
        enable_explainer_pass=bool(CONFIG["ENABLE_EXPLAINER_PASS"]),
        explainer_model=str(CONFIG["EXPLAINER_MODEL"]),
    )


def _prepare_image_store(args: SimpleNamespace) -> Optional[QuestionImageStore]:
    if not args.images_zip:
        return None
    if not os.path.exists(args.images_zip):
        raise FileNotFoundError(f"Fragenbilder-ZIP nicht gefunden: {args.images_zip}")
    return QuestionImageStore.from_zip(args.images_zip)


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
        schema_review = schema_review_pass(topic_keys)
        schema_reconstruction = schema_reconstruction_pass()
        schema_explainer = schema_explainer_pass()

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
        image_store = _prepare_image_store(args)
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
            topic_catalog=catalog,
            schema_a=schema_a,
            schema_b=schema_b,
            schema_review=schema_review,
            schema_reconstruction=schema_reconstruction,
            schema_explainer=schema_explainer,
            cleanup_spec=cleanup_spec,
            knowledge_base=knowledge_base,
            image_store=image_store,
            progress_callback=on_progress,
        )

        progress_bar.progress(1.0)
        st.success(f"Analyse beendet. Ergebnis gespeichert unter: {args.output}")

    except Exception as exc:
        st.exception(exc)


if __name__ == "__main__":
    main()
