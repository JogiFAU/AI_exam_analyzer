"""Streamlit UI for local execution of the AI exam analyzer."""

import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import streamlit as st

from ai_exam_analyzer.config import CONFIG
from ai_exam_analyzer.cost_tracking import format_eur
from ai_exam_analyzer.image_store import QuestionImageStore
from ai_exam_analyzer.io_utils import load_json, save_json
from ai_exam_analyzer.model_profiles import (
    QUALITY_PROFILE_LABELS,
    QUALITY_PROFILE_OPTIONS,
    get_quality_cost_profile,
)
from ai_exam_analyzer.knowledge_base import (
    build_knowledge_base_from_zip,
    load_index_json,
    save_index_json,
)
from ai_exam_analyzer.processor import process_questions
from ai_exam_analyzer.auto_tuning import recommend_settings
from ai_exam_analyzer.schemas import (
    schema_explainer_pass,
    schema_pass_a,
    schema_pass_b,
    schema_reconstruction_pass,
    schema_review_pass,
    schema_abstraction_cluster_refinement,
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




def _provider_ui_defaults(provider: str) -> Dict[str, Any]:
    p = (provider or "openai").strip().lower()
    if p == "gemini":
        return {
            "pass_a_temperature": 1.0,
            "pass_b_reasoning_effort": "medium",
            "trigger_answer_conf": 0.85,
            "trigger_topic_conf": 0.88,
            "apply_change_min_conf_b": 0.84,
            "low_conf_maintenance_threshold": 0.72,
        }
    return {
        "pass_a_temperature": float(CONFIG["PASSA_TEMPERATURE"]),
        "pass_b_reasoning_effort": CONFIG["PASSB_REASONING_EFFORT"],
        "trigger_answer_conf": float(CONFIG["TRIGGER_ANSWER_CONF"]),
        "trigger_topic_conf": float(CONFIG["TRIGGER_TOPIC_CONF"]),
        "apply_change_min_conf_b": float(CONFIG["APPLY_CHANGE_MIN_CONF_B"]),
        "low_conf_maintenance_threshold": float(CONFIG["LOW_CONF_MAINTENANCE_THRESHOLD"]),
    }

def _get_default_documents_dir() -> str:
    home_dir = os.path.expanduser("~")
    documents_dir = os.path.join(home_dir, "Documents")
    if os.path.isdir(documents_dir):
        return documents_dir
    return home_dir


def _profile_defaults_for_widgets(provider: str, profile_name: str) -> Dict[str, Any]:
    profile = get_quality_cost_profile(provider=provider, profile=profile_name)
    return {
        "pass_a_temperature": profile.pass_a_temperature,
        "pass_b_reasoning_effort": profile.pass_b_reasoning_effort,
        "trigger_answer_conf": profile.trigger_answer_conf,
        "trigger_topic_conf": profile.trigger_topic_conf,
        "apply_change_min_conf_b": profile.apply_change_min_conf_b,
        "low_conf_maintenance_threshold": profile.low_conf_maintenance_threshold,
        "knowledge_top_k": profile.knowledge_top_k,
        "knowledge_max_chars": profile.knowledge_max_chars,
        "knowledge_min_score": profile.knowledge_min_score,
        "enable_review_pass": profile.enable_review_pass,
        "enable_reconstruction_pass": profile.enable_reconstruction_pass,
        "enable_explainer_pass": profile.enable_explainer_pass,
    }


def _apply_settings_to_ui_state(settings: Dict[str, Any]) -> None:
    provider = str(settings.get("llm_provider") or st.session_state.get("llm_provider") or CONFIG["LLM_PROVIDER"])
    if provider not in {"openai", "gemini"}:
        provider = str(CONFIG["LLM_PROVIDER"])

    profile_name = str(settings.get("quality_cost_profile") or st.session_state.get("quality_cost_profile") or CONFIG["QUALITY_COST_PROFILE"])
    if profile_name not in QUALITY_PROFILE_OPTIONS:
        profile_name = str(CONFIG["QUALITY_COST_PROFILE"])

    st.session_state["llm_provider"] = provider
    st.session_state["quality_cost_profile"] = profile_name
    st.session_state[f"{provider}_last_applied_quality_cost_profile"] = profile_name

    merged = _profile_defaults_for_widgets(provider, profile_name)
    merged.update(settings)

    key_map = {
        "resume": "resume",
        "limit": "limit",
        "checkpoint_every": "checkpoint_every",
        "sleep": "sleep_seconds",
        "text_cluster_similarity": "text_cluster_similarity",
        "abstraction_cluster_similarity": "abstraction_cluster_similarity",
        "review_min_maintenance_severity": "review_min_maintenance_severity",
        "enable_repeat_reconstruction": "enable_repeat_reconstruction",
        "auto_apply_repeat_reconstruction": "auto_apply_repeat_reconstruction",
        "repeat_min_similarity": "repeat_min_similarity",
        "repeat_min_anchor_conf": "repeat_min_anchor_conf",
        "repeat_min_anchor_consensus": "repeat_min_anchor_consensus",
        "repeat_min_match_ratio": "repeat_min_match_ratio",
        "enable_explainer_pass": "enable_explainer_pass",
        "write_top_level": "write_top_level",
        "debug": "debug",
        "knowledge_subject_hint": "knowledge_subject_hint",
        "knowledge_chunk_chars": "knowledge_chunk_chars",
        "pass_a_temperature": f"{provider}_pass_a_temperature",
        "passA_temperature": f"{provider}_pass_a_temperature",
        "pass_b_reasoning_effort": f"{provider}_pass_b_reasoning_effort",
        "passB_reasoning_effort": f"{provider}_pass_b_reasoning_effort",
        "trigger_answer_conf": f"{provider}_trigger_answer_conf",
        "trigger_topic_conf": f"{provider}_trigger_topic_conf",
        "apply_change_min_conf_b": f"{provider}_apply_change_min_conf_b",
        "low_conf_maintenance_threshold": f"{provider}_low_conf_maintenance_threshold",
        "knowledge_top_k": f"{provider}_knowledge_top_k",
        "knowledge_max_chars": f"{provider}_knowledge_max_chars",
        "knowledge_min_score": f"{provider}_knowledge_min_score",
        "enable_review_pass": f"{provider}_enable_review_pass",
        "enable_reconstruction_pass": f"{provider}_enable_reconstruction_pass",
    }
    for cfg_key, state_key in key_map.items():
        if cfg_key in merged:
            st.session_state[state_key] = merged[cfg_key]
    if "only_question_ids" in merged and isinstance(merged["only_question_ids"], list):
        st.session_state["only_question_ids_raw"] = ", ".join(str(x) for x in merged["only_question_ids"])


def _sync_profile_defaults_when_changed(provider: str, profile_name: str) -> None:
    state_key = f"{provider}_last_applied_quality_cost_profile"
    previous = st.session_state.get(state_key)
    if previous == profile_name:
        return
    profile_defaults = _profile_defaults_for_widgets(provider, profile_name)
    key_map = {
        "pass_a_temperature": f"{provider}_pass_a_temperature",
        "pass_b_reasoning_effort": f"{provider}_pass_b_reasoning_effort",
        "trigger_answer_conf": f"{provider}_trigger_answer_conf",
        "trigger_topic_conf": f"{provider}_trigger_topic_conf",
        "apply_change_min_conf_b": f"{provider}_apply_change_min_conf_b",
        "low_conf_maintenance_threshold": f"{provider}_low_conf_maintenance_threshold",
        "knowledge_top_k": f"{provider}_knowledge_top_k",
        "knowledge_max_chars": f"{provider}_knowledge_max_chars",
        "knowledge_min_score": f"{provider}_knowledge_min_score",
        "enable_review_pass": f"{provider}_enable_review_pass",
        "enable_reconstruction_pass": f"{provider}_enable_reconstruction_pass",
        "enable_explainer_pass": "enable_explainer_pass",
    }
    for cfg_key, value in profile_defaults.items():
        state_target = key_map.get(cfg_key)
        if state_target:
            st.session_state[state_target] = value
    st.session_state[state_key] = profile_name


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

    pending_cfg = st.session_state.pop("_pending_analysis_config", None)
    if isinstance(pending_cfg, dict):
        _apply_settings_to_ui_state(pending_cfg)
        st.session_state["analysis_config_apply_success"] = True

    data_folder = st.session_state["data_folder"]
    output_folder = st.session_state["output_folder"]

    input_default_name = os.path.basename(CONFIG["INPUT_PATH"]) or "export.json"
    topics_default_name = os.path.basename(CONFIG["TOPICS_PATH"]) or "topic-tree.json"
    output_default_name = os.path.basename(CONFIG["OUTPUT_PATH"]) or ""
    images_zip_default_name = os.path.basename(CONFIG["IMAGES_ZIP_PATH"]) or "images.zip"
    knowledge_zip_default_name = os.path.basename(CONFIG["KNOWLEDGE_ZIP_PATH"]) or "knowledge.zip"
    knowledge_index_default_name = os.path.basename(CONFIG["KNOWLEDGE_INDEX_PATH"]) or "knowledge.index.json"

    with st.sidebar:
        st.header("Einstellungen")
        if st.session_state.pop("analysis_config_apply_success", False):
            st.success("Einstellungen aus der Analyse-Konfig wurden in die UI übernommen.")
        run_mode = st.radio(
            "Modus",
            options=["Prä-Analyse & Einstellungen", "Vollständige Analyse", "Postprocessing only", "Explainer only"],
            index=0,
            help="Prä-Analyse & Einstellungen ermittelt robuste Parameter vor der vollständigen Analyse. Postprocessing-only berechnet Review/Reconstruction auf bestehendem aiAudit neu; Explainer-only berechnet nur didaktische Erklärungen für bereits annotierte Daten.",
        )
        is_postprocess_only = (run_mode in {"Postprocessing only", "Explainer only"})
        is_explainer_only = (run_mode == "Explainer only")
        is_tuning_only = (run_mode == "Prä-Analyse & Einstellungen")
        is_full_analysis = (run_mode == "Vollständige Analyse")

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
            ]
            if not is_tuning_only:
                defaults.append(("Output", output_status_name))
            if is_full_analysis:
                defaults.append(("Bilder ZIP", images_zip_default_name))
            if is_full_analysis or is_tuning_only:
                defaults.append(("Knowledge ZIP", knowledge_zip_default_name))
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

            if is_tuning_only:
                output_path = output_default_path
            else:
                output_path = _file_picker_row(
                    state_key="output_file",
                    label="Output JSON",
                    default_path=output_default_path,
                    start_dir=output_folder,
                    help_text="Zieldatei für annotierte Ausgabe (wird automatisch erstellt).",
                    require_existing=False,
                )

            if is_tuning_only:
                only_question_ids_raw = ""
            else:
                only_question_ids_raw = st.text_input(
                    "Nur diese Frage-ID(s) verarbeiten (optional)",
                    value="",
                    key="only_question_ids_raw",
                    help="Kommagetrennte IDs; leer = alle Fragen. Wenige IDs sind hilfreich für Tests oder Nachläufe; leer verarbeitet den gesamten Datensatz.",
                )
            analysis_config_default_name = "analysis_config.json"
            analysis_config_path = _file_picker_row(
                state_key="analysis_config_file",
                label="Analyse-Konfig JSON",
                default_path=_resolve_path(folder=data_folder, filename=analysis_config_default_name),
                start_dir=data_folder,
                help_text="Konfig mit Parametern aus Prä-Analyse & Einstellungen. Sie wird erst durch 'Einstellungen anwenden' übernommen.",
                optional=True,
            )
            save_tuning_config_path = _file_picker_row(
                state_key="save_tuning_config_file",
                label="Speicherziel Parameter-Konfig",
                default_path=_resolve_path(folder=data_folder, filename=analysis_config_default_name),
                start_dir=data_folder,
                help_text="Zieldatei für ermittelte Parameter aus Prä-Analyse & Einstellungen.",
                optional=False,
                require_existing=False,
            ) if is_tuning_only else analysis_config_path

            if (not is_tuning_only) and (not is_postprocess_only):
                config_cols = st.columns([1, 3])
                with config_cols[0]:
                    apply_config = st.button(
                        "Einstellungen anwenden",
                        key="apply_analysis_config",
                        disabled=not (analysis_config_path and os.path.exists(analysis_config_path)),
                        help="Lädt die Analyse-Konfig aktiv in die UI. Danach können alle Werte weiter angepasst werden.",
                    )
                with config_cols[1]:
                    if analysis_config_path and os.path.exists(analysis_config_path):
                        st.caption("Konfig gefunden. Sie wird erst nach Klick auf **Einstellungen anwenden** übernommen.")
                    elif analysis_config_path:
                        st.caption("Konfig-Datei noch nicht gefunden.")
                if apply_config:
                    try:
                        loaded_cfg = load_json(analysis_config_path)
                        if not isinstance(loaded_cfg, dict):
                            raise ValueError("Analyse-Konfig muss ein JSON-Objekt sein.")
                        if isinstance(loaded_cfg.get("settings"), dict):
                            loaded_cfg = loaded_cfg["settings"]
                        st.session_state["_pending_analysis_config"] = loaded_cfg
                        st.session_state["analysis_config_applied_path"] = analysis_config_path
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Analyse-Konfig konnte nicht geladen werden: {exc}")

            cleanup_spec = ""

            if is_full_analysis:
                images_default_path = _resolve_path(folder=data_folder, filename=images_zip_default_name)
                images_default_exists = os.path.exists(images_default_path)
                use_images_zip = st.checkbox(
                    "Fragenbilder ZIP nutzen",
                    value=images_default_exists,
                    help="Wenn aktiv, werden Fragebilder aus einer ZIP geladen und dem Modell mitgegeben. Das ist nur in der vollständigen Analyse relevant; Postprocessing und Prä-Analyse & Einstellungen nutzen keine Fragebilder.",
                )
                if not images_default_exists:
                    st.caption("ℹ️ `images.zip` nicht im Input-Ordner gefunden – Nutzung standardmäßig aus, manuelle Auswahl weiterhin möglich.")
                images_zip = _file_picker_row(
                    state_key="images_zip_file",
                    label="Fragenbilder ZIP",
                    default_path=_resolve_path(folder=data_folder, filename=images_zip_default_name),
                    start_dir=data_folder,
                    help_text="ZIP mit Fragebildern (Dateinamen enthalten die Frage-ID).",
                    optional=True,
                ) if use_images_zip else ""
            else:
                images_zip = ""

            if is_full_analysis or is_tuning_only:
                knowledge_default_path = _resolve_path(folder=data_folder, filename=knowledge_zip_default_name)
                knowledge_default_exists = os.path.exists(knowledge_default_path)
                use_knowledge_zip = st.checkbox(
                    "Knowledge ZIP nutzen",
                    value=knowledge_default_exists,
                    help="Wenn aktiv, wird Wissen aus einer ZIP-Datei geladen. Relevant für vollständige Analyse und Prä-Analyse & Einstellungen; reine Postprocessing-Modi verwenden bestehende aiAudit-Daten.",
                )
                if not knowledge_default_exists:
                    st.caption("ℹ️ `knowledge.zip` nicht im Input-Ordner gefunden – Nutzung standardmäßig aus, manuelle Auswahl weiterhin möglich.")
                knowledge_zip = _file_picker_row(
                    state_key="knowledge_zip_file",
                    label="Knowledge ZIP",
                    default_path=_resolve_path(folder=data_folder, filename=knowledge_zip_default_name),
                    start_dir=data_folder,
                    help_text="ZIP mit Wissensdokumenten (PDF/TXT/MD).",
                    optional=True,
                ) if use_knowledge_zip else ""

                knowledge_index_default_path = _resolve_path(folder=data_folder, filename=knowledge_index_default_name)
                knowledge_index_default_exists = os.path.exists(knowledge_index_default_path)
                use_knowledge_index = st.checkbox(
                    "Knowledge-Index nutzen",
                    value=knowledge_index_default_exists,
                    help="Optionaler Index für schnelleren Start; wird geladen/geschrieben. Nicht relevant für reine Postprocessing-Modi.",
                )
                knowledge_index = _file_picker_row(
                    state_key="knowledge_index_file",
                    label="Knowledge Index JSON",
                    default_path=_resolve_path(folder=data_folder, filename=knowledge_index_default_name),
                    start_dir=data_folder,
                    help_text="Optionaler Index-Cache als JSON.",
                    optional=True,
                ) if use_knowledge_index else ""
            else:
                knowledge_zip = ""
                knowledge_index = ""

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
            llm_provider = st.selectbox(
                "LLM Provider",
                options=["openai", "gemini"],
                index=0 if CONFIG["LLM_PROVIDER"] == "openai" else 1,
                key="llm_provider",
                help="Wählt den Modellanbieter für alle KI-Schritte. OpenAI und Gemini verwenden unterschiedliche Modellnamen, Kostenstrukturen und Kontextfenster. Ein Wechsel setzt provider-spezifische Profildefaults; danach können die angezeigten Parameter weiter angepasst werden.",
            )
            key_env = "OPENAI_API_KEY" if llm_provider == "openai" else "GEMINI_API_KEY"
            api_key_value = os.getenv(key_env, "")
            api_key = st.text_input(
                f"{llm_provider.title()} API Key",
                type="password",
                value=api_key_value,
                help="API-Key für den Zugriff auf den gewählten Provider. Der Key wird für diese Sitzung als Umgebungsvariable gesetzt, aber nicht in gespeicherte UI-Konfigurationen geschrieben.",
            )
            profile_default = str(CONFIG.get("QUALITY_COST_PROFILE", "quality"))
            profile_default_index = QUALITY_PROFILE_OPTIONS.index(profile_default) if profile_default in QUALITY_PROFILE_OPTIONS else 1
            quality_cost_profile = st.selectbox(
                "Priorität",
                options=QUALITY_PROFILE_OPTIONS,
                index=profile_default_index,
                key="quality_cost_profile",
                format_func=lambda value: QUALITY_PROFILE_LABELS.get(value, value),
                help="Wählt ein abgestimmtes Start-Set aus Modellen, Reasoning-Aufwand, Confidence-Schwellen, Knowledge-Budget und optionalen teuren Schritten. Richtung Qualität nutzt stärkere Modelle und mehr Kontext, kostet aber mehr. Richtung Kostenoptimierung reduziert Modellstärke, Reasoning und Kontext; danach bleiben die Detailwerte manuell anpassbar.",
            )
            _sync_profile_defaults_when_changed(llm_provider, quality_cost_profile)
            selected_profile = get_quality_cost_profile(provider=llm_provider, profile=quality_cost_profile)
            default_pass_a_model = selected_profile.pass_a_model
            default_pass_b_model = selected_profile.pass_b_model
            default_review_model = selected_profile.review_model
            default_reconstruction_model = selected_profile.reconstruction_model
            default_explainer_model = selected_profile.explainer_model
            provider_defaults = {
                "pass_a_temperature": selected_profile.pass_a_temperature,
                "pass_b_reasoning_effort": selected_profile.pass_b_reasoning_effort,
                "trigger_answer_conf": selected_profile.trigger_answer_conf,
                "trigger_topic_conf": selected_profile.trigger_topic_conf,
                "apply_change_min_conf_b": selected_profile.apply_change_min_conf_b,
                "low_conf_maintenance_threshold": selected_profile.low_conf_maintenance_threshold,
            }
            kb_budget_defaults = selected_profile
            st.caption(
                f"Aktives Set: Pass A `{default_pass_a_model}`, Pass B `{default_pass_b_model}`, "
                f"Review/Reconstruction `{default_reconstruction_model}`. "
                "Für Feinsteuerung später kann diese Auswahl erweitert werden."
            )
        auto_dataset_tuning = bool(is_tuning_only)

        checkpoint_every = int(CONFIG["CHECKPOINT_EVERY"])
        text_cluster_similarity = float(CONFIG["TEXT_CLUSTER_SIMILARITY"])
        abstraction_cluster_similarity = float(CONFIG["ABSTRACTION_CLUSTER_SIMILARITY"])
        enable_review_pass = False if is_explainer_only else bool(selected_profile.enable_review_pass)
        review_min_maintenance_severity = int(CONFIG["REVIEW_MIN_MAINTENANCE_SEVERITY"])
        enable_reconstruction_pass = False if is_explainer_only else bool(selected_profile.enable_reconstruction_pass)
        force_rerun_review = False
        force_rerun_reconstruction = False
        force_rerun_explainer = False
        resume = bool(CONFIG["RESUME"])
        limit = int(CONFIG["LIMIT"])
        sleep_seconds = float(CONFIG["SLEEP"])
        pass_a_model = str(default_pass_a_model)
        pass_b_model = str(default_pass_b_model)
        review_model = str(default_review_model)
        reconstruction_model = str(default_reconstruction_model)
        pass_a_temperature = float(provider_defaults["pass_a_temperature"])
        pass_b_reasoning_effort = str(provider_defaults["pass_b_reasoning_effort"])
        trigger_answer_conf = float(provider_defaults["trigger_answer_conf"])
        trigger_topic_conf = float(provider_defaults["trigger_topic_conf"])
        apply_change_min_conf_b = float(provider_defaults["apply_change_min_conf_b"])
        low_conf_maintenance_threshold = float(provider_defaults["low_conf_maintenance_threshold"])
        enable_repeat_reconstruction = bool(CONFIG["ENABLE_REPEAT_RECONSTRUCTION"])
        auto_apply_repeat_reconstruction = bool(CONFIG["AUTO_APPLY_REPEAT_RECONSTRUCTION"])
        repeat_min_similarity = float(CONFIG["REPEAT_MIN_SIMILARITY"])
        repeat_min_anchor_conf = float(CONFIG["REPEAT_MIN_ANCHOR_CONF"])
        repeat_min_anchor_consensus = int(CONFIG["REPEAT_MIN_ANCHOR_CONSENSUS"])
        repeat_min_match_ratio = float(CONFIG["REPEAT_MIN_MATCH_RATIO"])
        enable_explainer_pass = bool(selected_profile.enable_explainer_pass)
        explainer_model = str(default_explainer_model)
        write_top_level = bool(CONFIG["WRITE_TOP_LEVEL"])
        debug = bool(CONFIG["DEBUG"])

        if is_tuning_only:
            st.info("Prä-Analyse & Einstellungen: Es werden nur Datenquellen, API und Knowledge-Base angezeigt. Die Detailparameter werden durch die Analyse ermittelt und anschließend als Konfig gespeichert.")
            enable_review_pass = bool(selected_profile.enable_review_pass)
            enable_reconstruction_pass = False
            enable_explainer_pass = True
        elif is_explainer_only:
            with st.expander("💬 Explainer", expanded=True):
                enable_review_pass = False
                enable_reconstruction_pass = False
                enable_explainer_pass = True
                checkpoint_every = st.number_input(
                    "Checkpoint alle N Fragen",
                    min_value=1,
                    value=int(CONFIG["CHECKPOINT_EVERY"]),
                    key="checkpoint_every",
                    help="Auch im Explainer-only-Modus werden Zwischenergebnisse geschrieben. Niedrigere Werte reduzieren Datenverlust bei Abbruch, höhere Werte schreiben seltener.",
                )
                text_cluster_similarity = st.slider(
                    "Question-Cluster Similarity",
                    0.0,
                    1.0,
                    float(CONFIG["TEXT_CLUSTER_SIMILARITY"]),
                    0.01,
                    key="text_cluster_similarity",
                    help="Wird im Postprocessing-Kontext zur Aktualisierung von Frage-Clustern verwendet. Niedriger gruppiert mehr, höher ist strenger.",
                )
                abstraction_cluster_similarity = st.slider(
                    "Abstraction-Cluster Similarity",
                    0.0,
                    1.0,
                    float(CONFIG["ABSTRACTION_CLUSTER_SIMILARITY"]),
                    0.01,
                    key="abstraction_cluster_similarity",
                    help="Wird am Ende des Postprocessing-Laufs für Abstraktionscluster genutzt. Niedriger gruppiert breiter, höher trennt stärker.",
                )
                force_rerun_explainer = st.checkbox(
                    "Explainer immer neu berechnen",
                    value=True,
                    help="Erzwingt eine neue didaktische Erklärung für jede verarbeitete Frage. Aktiv ist im Explainer-only-Modus meist sinnvoll, weil genau dieser Schritt nachgezogen werden soll; deaktiviert würde vorhandene Erklärungen wiederverwenden.",
                )
                explainer_model = str(default_explainer_model)
                st.caption(f"Explainer Modell: `{explainer_model}`")
                write_top_level = st.checkbox(
                    "Top-Level ai* Felder schreiben",
                    value=CONFIG["WRITE_TOP_LEVEL"],
                    key="write_top_level",
                    help="Wird auch in Postprocessing-Läufen angewendet. Aktiv aktualisiert praktische ai*-Kurzfelder auf Fragenebene; deaktiviert verändert nur aiAudit und hält den Export schlanker.",
                )
        elif is_postprocess_only:
            with st.expander("⚙️ Postprocessing", expanded=True):
                st.caption("Angezeigt werden nur die im Postprocessing tatsächlich verwendeten Optionen: Checkpoints, Cluster-Aktualisierung, Review, Reconstruction, optional Explainer und Top-Level-Ausgabe.")
                checkpoint_every = st.number_input(
                    "Checkpoint alle N Fragen",
                    min_value=1,
                    value=int(CONFIG["CHECKPOINT_EVERY"]),
                    key="checkpoint_every",
                    help="Postprocessing speichert ebenfalls Zwischenergebnisse. Niedrigere Werte reduzieren Datenverlust bei Abbruch, höhere Werte schreiben seltener.",
                )
                text_cluster_similarity = st.slider(
                    "Question-Cluster Similarity",
                    0.0,
                    1.0,
                    float(CONFIG["TEXT_CLUSTER_SIMILARITY"]),
                    0.01,
                    key="text_cluster_similarity",
                    help="Postprocessing aktualisiert Frage-Cluster. Niedrigere Werte gruppieren mehr Fragen zusammen, höhere Werte sind konservativer.",
                )
                abstraction_cluster_similarity = st.slider(
                    "Abstraction-Cluster Similarity",
                    0.0,
                    1.0,
                    float(CONFIG["ABSTRACTION_CLUSTER_SIMILARITY"]),
                    0.01,
                    key="abstraction_cluster_similarity",
                    help="Postprocessing aktualisiert Abstraktionscluster. Niedrigere Werte erlauben breitere Gruppen, höhere Werte trennen stärker.",
                )
                enable_review_pass = st.checkbox(
                    "Pass C (Deep Review) aktivieren",
                    value=bool(selected_profile.enable_review_pass),
                    key=f"{llm_provider}_enable_review_pass",
                    help="Optionaler dritter Review-Pass für wartungsintensive Fragen. Aktiv prüft bestehende aiAudit-Ergebnisse gründlicher, verursacht aber zusätzliche Modellkosten.",
                )
                if enable_review_pass:
                    st.caption(f"Pass C Modell: `{default_review_model}`")
                    review_min_maintenance_severity = st.select_slider(
                        "Pass C ab Wartungs-Severity",
                        options=[1, 2, 3],
                        value=int(CONFIG["REVIEW_MIN_MAINTENANCE_SEVERITY"]),
                        key="review_min_maintenance_severity",
                        help="Pass C läuft nur ab diesem Wartungs-Schweregrad. Niedrigere Werte prüfen mehr Fragen gründlich und erhöhen Qualität/Kosten. Höhere Werte beschränken den teuren Review auf kritischere Fälle.",
                    )
                    force_rerun_review = st.checkbox(
                        "Review immer neu berechnen",
                        value=False,
                        help="Erzwingt eine Neuberechnung des Review-Passes, auch wenn bereits ein Ergebnis vorliegt. Aktiv ist sinnvoll nach Parameter-/Modelländerungen, kostet aber zusätzliche API-Aufrufe.",
                    )
                enable_reconstruction_pass = st.checkbox(
                    "Reconstruction-Pass aktivieren",
                    value=bool(selected_profile.enable_reconstruction_pass),
                    key=f"{llm_provider}_enable_reconstruction_pass",
                    help="Führt eine Rekonstruktions-/Altfrage-Bewertung pro Frage aus. Aktiv aktualisiert diese Audit-Sektion, deaktiviert blendet sie im Postprocessing aus und spart Kosten.",
                )
                if enable_reconstruction_pass:
                    st.caption(f"Reconstruction Modell: `{default_reconstruction_model}`")
                    force_rerun_reconstruction = st.checkbox(
                        "Reconstruction immer neu berechnen",
                        value=False,
                        help="Erzwingt eine neue Reconstruction-Bewertung. Aktiv aktualisiert alte Ergebnisse zuverlässig; deaktiviert berechnet nur fehlende/fehlerhafte Einträge.",
                    )
                enable_explainer_pass = st.checkbox(
                    "Explainer-Pass aktivieren",
                    value=bool(selected_profile.enable_explainer_pass),
                    key="enable_explainer_pass",
                    help="Erzeugt didaktische Erklärungen auf bestehendem aiAudit. Aktiv verbessert Nachvollziehbarkeit, erzeugt aber zusätzliche Modellkosten.",
                )
                if enable_explainer_pass:
                    explainer_model = str(default_explainer_model)
                    st.caption(f"Explainer Modell: `{explainer_model}`")
                    force_rerun_explainer = st.checkbox(
                        "Explainer immer neu berechnen",
                        value=False,
                        help="Erzwingt eine neue didaktische Erklärung. Aktiv ist sinnvoll bei geänderten Modellen oder Promptlogik, erhöht aber Kosten; deaktiviert erhält vorhandene Erklärungen.",
                    )
                write_top_level = st.checkbox(
                    "Top-Level ai* Felder schreiben",
                    value=CONFIG["WRITE_TOP_LEVEL"],
                    key="write_top_level",
                    help="Postprocessing kann ai*-Kurzfelder aus dem aktualisierten aiAudit neu schreiben. Aktiv erleichtert Weiterverarbeitung; deaktiviert belässt Änderungen primär im aiAudit.",
                )
        else:
            with st.expander("⚙️ Pipeline", expanded=False):
                checkpoint_every = st.number_input(
                    "Checkpoint alle N Fragen",
                    min_value=1,
                    value=int(CONFIG["CHECKPOINT_EVERY"]),
                    key="checkpoint_every",
                    help="Speichert regelmäßig Zwischenergebnisse. Niedrigere Werte reduzieren Datenverlust bei Abbruch, erzeugen aber mehr Schreibzugriffe. Höhere Werte sind etwas schneller, riskieren aber größere Wiederholungen nach Fehlern.",
                )
                text_cluster_similarity = st.slider(
                    "Question-Cluster Similarity",
                    0.0,
                    1.0,
                    float(CONFIG["TEXT_CLUSTER_SIMILARITY"]),
                    0.01,
                    key="text_cluster_similarity",
                    help="Ähnlichkeitsschwelle für inhaltliche Frage-Cluster. Niedrigere Werte gruppieren mehr Fragen zusammen und können Wiederholungen stärker nutzen, riskieren aber falsche Cluster. Höhere Werte sind strenger und sicherer, finden aber weniger verwandte Fragen.",
                )
                abstraction_cluster_similarity = st.slider(
                    "Abstraction-Cluster Similarity",
                    0.0,
                    1.0,
                    float(CONFIG["ABSTRACTION_CLUSTER_SIMILARITY"]),
                    0.01,
                    key="abstraction_cluster_similarity",
                    help="Ähnlichkeitsschwelle für Cluster der abstrahierten Fragen. Niedrigere Werte erlauben breitere thematische Gruppen; höhere Werte halten Cluster enger und reduzieren falsch zusammengeführte Themen.",
                )
                enable_review_pass = st.checkbox(
                    "Pass C (Deep Review) aktivieren",
                    value=bool(selected_profile.enable_review_pass),
                    key=f"{llm_provider}_enable_review_pass",
                    help="Optionaler dritter Review-Pass für wartungsintensive Fragen. Das Prioritätsprofil setzt nur den Startwert.",
                )
                review_model = str(default_review_model)
                st.caption(f"Pass C Modell: `{review_model}`")
                review_min_maintenance_severity = st.select_slider(
                    "Pass C ab Wartungs-Severity",
                    options=[1, 2, 3],
                    value=int(CONFIG["REVIEW_MIN_MAINTENANCE_SEVERITY"]),
                    key="review_min_maintenance_severity",
                    help="Pass C läuft nur ab diesem Wartungs-Schweregrad. Niedrigere Werte prüfen mehr Fragen gründlich und erhöhen Qualität/Kosten. Höhere Werte beschränken den teuren Review auf kritischere Fälle.",
                    disabled=not enable_review_pass,
                )
                enable_reconstruction_pass = st.checkbox(
                    "Reconstruction-Pass aktivieren",
                    value=bool(selected_profile.enable_reconstruction_pass),
                    key=f"{llm_provider}_enable_reconstruction_pass",
                    help="Führt eine Rekonstruktions-/Altfrage-Bewertung pro Frage aus und annotiert das Ergebnis. Das Prioritätsprofil setzt nur den Startwert.",
                )
                reconstruction_model = str(default_reconstruction_model)
                st.caption(f"Reconstruction Modell: `{reconstruction_model}`")
                resume = st.checkbox("Resume aktiv", value=CONFIG["RESUME"], key="resume", help="Überspringt bereits abgeschlossene Fragen mit passender Pipeline-Version. Aktiv spart Kosten bei Fortsetzungen; deaktiviert erzwingt eine vollständige Neuberechnung und kann bestehende KI-Annotationen aktualisieren.")
                limit = st.number_input("Limit (0 = alle Fragen)", min_value=0, value=int(CONFIG["LIMIT"]), key="limit", help="Begrenzt die Anzahl verarbeiteter Fragen. 0 verarbeitet alles. Kleine Werte eignen sich für kostengünstige Testläufe; höhere Werte bzw. 0 führen den kompletten Workflow aus.")
                sleep_seconds = st.number_input(
                    "Pause je Frage (Sek.)",
                    min_value=0.0,
                    value=float(CONFIG["SLEEP"]),
                    step=0.05,
                    key="sleep_seconds",
                    help="Kurze Pause zwischen zwei API-Aufrufen. Höhere Werte schonen Rate-Limits und reduzieren temporäre API-Fehler, verlängern aber die Laufzeit. Niedrigere Werte sind schneller, können bei großen Datensätzen aber eher Rate-Limits treffen.",
                )
                pass_a_model = str(default_pass_a_model)
                pass_b_model = str(default_pass_b_model)
                st.caption(f"Pass A Modell: `{pass_a_model}`")
                st.caption(f"Pass B Modell: `{pass_b_model}`")
                pass_a_temperature = st.number_input(
                    "Pass A Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=float(provider_defaults["pass_a_temperature"]),
                    key=f"{llm_provider}_pass_a_temperature",
                    step=0.1,
                    help="Sampling-Temperatur für Pass A. Das Prioritätsprofil setzt nur den Startwert.",
                )
                pass_b_reasoning_effort = st.selectbox(
                    "Pass B Reasoning Effort",
                    options=["low", "medium", "high", "xhigh"],
                    index=["low", "medium", "high", "xhigh"].index(str(provider_defaults["pass_b_reasoning_effort"])),
                    key=f"{llm_provider}_pass_b_reasoning_effort",
                    help="Rechenaufwand für Pass B. Das Prioritätsprofil setzt nur den Startwert.",
                )
                trigger_answer_conf = st.slider("Pass B Trigger: Answer Confidence", 0.0, 1.0, float(provider_defaults["trigger_answer_conf"]), 0.01, key=f"{llm_provider}_trigger_answer_conf", help="Antwort-Confidence unterhalb dieser Schwelle löst Pass B aus. Niedrigere Werte sparen Kosten, weil weniger Fälle verifiziert werden, riskieren aber unerkannte Fehler. Höhere Werte prüfen mehr unsichere Antworten und verbessern Qualität auf Kosten zusätzlicher API-Aufrufe.")
                trigger_topic_conf = st.slider("Pass B Trigger: Topic Confidence", 0.0, 1.0, float(provider_defaults["trigger_topic_conf"]), 0.01, key=f"{llm_provider}_trigger_topic_conf", help="Topic-Confidence unterhalb dieser Schwelle löst Pass B aus. Niedriger ist kostenorientierter und akzeptiert mehr Pass-A-Zuordnungen. Höher ist qualitätsorientierter und überprüft mehr potenziell falsche Topic-Zuweisungen.")
                apply_change_min_conf_b = st.slider("Änderung anwenden ab Pass-B Confidence", 0.0, 1.0, float(provider_defaults["apply_change_min_conf_b"]), 0.01, key=f"{llm_provider}_apply_change_min_conf_b", help="Mindestvertrauen, ab dem Pass-B-Korrekturen automatisch übernommen werden. Niedrigere Werte übernehmen mehr Änderungen, auch riskantere. Höhere Werte sind konservativer und lassen zweifelhafte Änderungen eher als Audit-Hinweis stehen.")
                low_conf_maintenance_threshold = st.slider("Wartung markieren unter Confidence", 0.0, 1.0, float(provider_defaults["low_conf_maintenance_threshold"]), 0.01, key=f"{llm_provider}_low_conf_maintenance_threshold", help="Unterhalb dieser Gesamt-Confidence wird eine Frage als Wartungskandidat markiert. Niedrigere Werte erzeugen weniger Warnungen, können Problemfälle übersehen. Höhere Werte markieren mehr Fragen zur Prüfung und erhöhen die Review-Last.")
                enable_repeat_reconstruction = st.checkbox(
                    "Repeat-Reconstruction aktivieren",
                    value=bool(CONFIG["ENABLE_REPEAT_RECONSTRUCTION"]),
                    key="enable_repeat_reconstruction",
                    help="Erkennt wiederholte Fragen über Jahrgänge und ergänzt entsprechende Audit-Signale. Aktiv kann Qualität verbessern und Kosten sparen, weil Muster genutzt werden. Deaktiviert vermeidet falsche Wiederholungsannahmen bei sehr heterogenen Datensätzen.",
                )
                auto_apply_repeat_reconstruction = st.checkbox(
                    "Repeat-Reconstruction Auto-Apply (nur Audit-Suggestion)",
                    value=bool(CONFIG["AUTO_APPLY_REPEAT_RECONSTRUCTION"]),
                    key="auto_apply_repeat_reconstruction",
                    help="Wendet sichere Repeat-Reconstruction-Vorschläge automatisch als Audit-Suggestion an. Aktiv spart manuelle Prüfung bei klaren Wiederholungen; deaktiviert hält alle Vorschläge rein informativ.",
                    disabled=(not enable_repeat_reconstruction),
                )
                repeat_min_similarity = st.slider("Repeat: Min Similarity", 0.0, 1.0, float(CONFIG["REPEAT_MIN_SIMILARITY"]), 0.01, key="repeat_min_similarity", help="Mindestähnlichkeit, ab der Fragen als Wiederholungs-Kandidaten gelten. Niedrigere Werte finden mehr Kandidaten, riskieren aber falsche Matches. Höhere Werte sind sicherer, übersehen aber abgewandelte Wiederholungen.", disabled=(not enable_repeat_reconstruction))
                repeat_min_anchor_conf = st.slider("Repeat: Min Anchor Confidence", 0.0, 1.0, float(CONFIG["REPEAT_MIN_ANCHOR_CONF"]), 0.01, key="repeat_min_anchor_conf", help="Mindestvertrauen für Ankerfragen, deren bekannte Bewertung Wiederholungen stützen darf. Niedriger nutzt mehr Anker, aber mit höherem Fehlerrisiko. Höher nutzt nur sehr sichere Anker und ist konservativer.", disabled=(not enable_repeat_reconstruction))
                repeat_min_anchor_consensus = st.number_input("Repeat: Min Anchor Consensus", min_value=1, value=int(CONFIG["REPEAT_MIN_ANCHOR_CONSENSUS"]), step=1, key="repeat_min_anchor_consensus", help="Mindestanzahl unabhängiger Anker, die dieselbe Richtung stützen müssen. Niedrigere Werte sind sensitiver und günstiger; höhere Werte erhöhen Sicherheit, benötigen aber mehr passende Wiederholungen.", disabled=(not enable_repeat_reconstruction))
                repeat_min_match_ratio = st.slider("Repeat: Min Match Ratio", 0.0, 1.0, float(CONFIG["REPEAT_MIN_MATCH_RATIO"]), 0.01, key="repeat_min_match_ratio", help="Mindestüberlappung zwischen Antworttexten von Anker und Ziel. Niedriger toleriert stärkere Umformulierungen, höher verlangt nahezu identische Antwortoptionen und reduziert Fehlübernahmen.", disabled=(not enable_repeat_reconstruction))
                enable_explainer_pass = st.checkbox(
                    "Explainer-Pass aktivieren",
                    value=bool(selected_profile.enable_explainer_pass),
                    key="enable_explainer_pass",
                    help="Erzeugt eine didaktische Erklärung pro Frage im Audit. Aktiv liefert bessere Nachvollziehbarkeit für Lern-/Review-Zwecke, verursacht aber zusätzliche Modellkosten. Deaktiviert spart Kosten und Laufzeit.",
                )
                explainer_model = str(default_explainer_model)
                st.caption(f"Explainer Modell: `{explainer_model}`")
                write_top_level = st.checkbox(
                    "Top-Level ai* Felder schreiben",
                    value=CONFIG["WRITE_TOP_LEVEL"],
                    key="write_top_level",
                    help="Schreibt zusätzliche ai*-Felder direkt in jede Frage. Aktiv erleichtert Export/Weiterverarbeitung. Deaktiviert hält die Ausgabe schlanker und belässt Details primär im aiAudit.",
                )
                debug = st.checkbox(
                    "Debug-Rohdaten speichern",
                    value=CONFIG["DEBUG"],
                    key="debug",
                    help="Speichert detaillierte Rohantworten unter aiAudit._debug. Aktiv hilft bei Fehlersuche und Qualitätsprüfung, vergrößert aber Ausgaben und kann sensible Prompt-/Antwortdetails enthalten. Deaktiviert ist schlanker.",
                )

        knowledge_subject_hint = str(st.session_state.get("knowledge_subject_hint", subject_hint_default)).strip()
        knowledge_top_k = int(kb_budget_defaults.knowledge_top_k)
        knowledge_max_chars = int(kb_budget_defaults.knowledge_max_chars)
        knowledge_min_score = float(kb_budget_defaults.knowledge_min_score)
        knowledge_chunk_chars = int(CONFIG["KNOWLEDGE_CHUNK_CHARS"])

        if is_full_analysis or is_tuning_only:
            with st.expander("🧠 Knowledge Base", expanded=False):
                knowledge_subject_hint = st.text_input(
                    "Subject Hint",
                    key="knowledge_subject_hint",
                    help="Fach-/Themenhinweis für die Knowledge-Base-Suche. Ein präziser Hinweis kann Retrieval-Treffer verbessern. Ein falscher oder zu enger Hinweis kann relevante Belege verdrängen; leer nutzt automatische Ableitung aus dem Topic-Tree.",
                )
                if is_full_analysis:
                    knowledge_top_k = st.number_input(
                        "Knowledge Top-K",
                        min_value=1,
                        value=int(kb_budget_defaults.knowledge_top_k),
                        key=f"{llm_provider}_knowledge_top_k",
                        help="Anzahl der Knowledge-Belege pro Frage. Höhere Werte geben dem Modell mehr Kontext und können die fachliche Sicherheit erhöhen, vergrößern aber Prompts und Kosten. Niedrigere Werte sparen Tokens und reduzieren Rauschen, können aber relevante Belege auslassen.",
                    )
                    knowledge_max_chars = st.number_input(
                        "Knowledge Max Chars",
                        min_value=500,
                        value=int(kb_budget_defaults.knowledge_max_chars),
                        key=f"{llm_provider}_knowledge_max_chars",
                        step=100,
                        help="Maximale Gesamtlänge aller Knowledge-Belege pro Frage. Höhere Werte erlauben ausführlicheren Kontext, erhöhen aber Tokenverbrauch und potenziell Ablenkung. Niedrigere Werte sind günstiger und fokussierter, riskieren aber abgeschnittene Begründungen.",
                    )
                    knowledge_min_score = st.slider(
                        "Knowledge Min Score",
                        0.0,
                        1.0,
                        float(kb_budget_defaults.knowledge_min_score),
                        0.01,
                        help="Mindestrelevanz eines Knowledge-Chunks. Niedrigere Werte geben mehr, aber potenziell schwächere Belege an das Modell. Höhere Werte reduzieren Kontext/Kosten und Rauschen, können aber hilfreiche Belege ausschließen.",
                        key=f"{llm_provider}_knowledge_min_score",
                    )
                else:
                    st.caption("Prä-Analyse & Einstellungen nutzt die Knowledge Base zur Analyse, zeigt aber keine Detailparameter an, weil diese vom Tuning-Lauf ermittelt und gespeichert werden.")
                knowledge_chunk_chars = st.number_input(
                    "Knowledge Chunk Chars",
                    min_value=200,
                    value=int(CONFIG["KNOWLEDGE_CHUNK_CHARS"]),
                    step=100,
                    key="knowledge_chunk_chars",
                    help="Chunk-Größe beim Parsen der Knowledge-ZIP. Kleinere Chunks erlauben präzisere Treffer, können aber Zusammenhänge zerlegen. Größere Chunks behalten Kontext, erhöhen jedoch Prompt-Länge und Kosten pro Treffer.",
                )

        if is_full_analysis:
            with st.expander("💾 Einstellungen speichern", expanded=False):
                save_ui_config_path = _file_picker_row(
                    state_key="save_ui_config_file",
                    label="Speicherziel UI-Konfig",
                    default_path=_resolve_path(folder=data_folder, filename=analysis_config_default_name),
                    start_dir=data_folder,
                    help_text="JSON-Datei, in die die aktuell in der UI sichtbaren Workflow-Einstellungen gespeichert werden. API-Keys werden bewusst nicht gespeichert.",
                    optional=False,
                    require_existing=False,
                )
                current_settings_payload = {
                    "llm_provider": llm_provider,
                    "quality_cost_profile": quality_cost_profile,
                    "resume": bool(resume),
                    "limit": int(limit),
                    "checkpoint_every": int(checkpoint_every),
                    "sleep": float(sleep_seconds),
                    "passA_temperature": float(pass_a_temperature),
                    "passB_reasoning_effort": str(pass_b_reasoning_effort),
                    "trigger_answer_conf": float(trigger_answer_conf),
                    "trigger_topic_conf": float(trigger_topic_conf),
                    "apply_change_min_conf_b": float(apply_change_min_conf_b),
                    "low_conf_maintenance_threshold": float(low_conf_maintenance_threshold),
                    "text_cluster_similarity": float(text_cluster_similarity),
                    "abstraction_cluster_similarity": float(abstraction_cluster_similarity),
                    "enable_review_pass": bool(enable_review_pass),
                    "review_min_maintenance_severity": int(review_min_maintenance_severity),
                    "enable_reconstruction_pass": bool(enable_reconstruction_pass),
                    "enable_repeat_reconstruction": bool(enable_repeat_reconstruction),
                    "auto_apply_repeat_reconstruction": bool(auto_apply_repeat_reconstruction),
                    "repeat_min_similarity": float(repeat_min_similarity),
                    "repeat_min_anchor_conf": float(repeat_min_anchor_conf),
                    "repeat_min_anchor_consensus": int(repeat_min_anchor_consensus),
                    "repeat_min_match_ratio": float(repeat_min_match_ratio),
                    "enable_explainer_pass": bool(enable_explainer_pass),
                    "write_top_level": bool(write_top_level),
                    "debug": bool(debug),
                    "knowledge_subject_hint": knowledge_subject_hint.strip(),
                    "knowledge_top_k": int(knowledge_top_k),
                    "knowledge_max_chars": int(knowledge_max_chars),
                    "knowledge_min_score": float(knowledge_min_score),
                    "knowledge_chunk_chars": int(knowledge_chunk_chars),
                    "only_question_ids": [x.strip() for x in (only_question_ids_raw or "").split(",") if x.strip()],
                }
                if st.button(
                    "Aktuelle Einstellungen speichern",
                    key="save_current_ui_config",
                    disabled=not bool(save_ui_config_path.strip()),
                    help="Speichert genau die aktuell sichtbaren UI-Werte in die angegebene JSON-Datei. Diese Datei kann später über 'Einstellungen anwenden' wieder geladen und danach erneut manuell verändert werden.",
                ):
                    try:
                        save_json(save_ui_config_path.strip(), current_settings_payload)
                        st.success(f"Einstellungen gespeichert: {save_ui_config_path}")
                    except Exception as exc:
                        st.error(f"Einstellungen konnten nicht gespeichert werden: {exc}")

    return SimpleNamespace(
        postprocess_only=bool(is_postprocess_only),
        tuning_only=bool(is_tuning_only),
        force_rerun_review=bool(force_rerun_review),
        force_rerun_reconstruction=bool(force_rerun_reconstruction),
        force_rerun_explainer=bool(force_rerun_explainer),
        only_question_ids=[x.strip() for x in (only_question_ids_raw or "").split(",") if x.strip()],
        input=input_path,
        topics=topics_path,
        output=(output_path or _derive_output_path_from_input(input_path, output_folder)),
        api_key=api_key,
        llm_provider=llm_provider,
        quality_cost_profile=quality_cost_profile,
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
        cost_report=str(CONFIG.get("COST_REPORT_PATH", "")),
        topic_candidate_outside_force_passb_conf=float(CONFIG["TOPIC_CANDIDATE_OUTSIDE_FORCE_PASSB_CONF"]),
        enable_repeat_reconstruction=bool(enable_repeat_reconstruction),
        auto_apply_repeat_reconstruction=bool(auto_apply_repeat_reconstruction),
        repeat_min_similarity=float(repeat_min_similarity),
        repeat_min_anchor_conf=float(repeat_min_anchor_conf),
        repeat_min_anchor_consensus=int(repeat_min_anchor_consensus),
        repeat_min_match_ratio=float(repeat_min_match_ratio),
        enable_reconstruction_pass=bool(enable_reconstruction_pass),
        reconstruction_model=reconstruction_model.strip(),
        enable_explainer_pass=bool(enable_explainer_pass),
        explainer_model=explainer_model.strip(),
        auto_dataset_tuning=bool(auto_dataset_tuning),
        analysis_config_path=(analysis_config_path or "").strip(),
        save_tuning_config_path=(save_tuning_config_path or "").strip(),
    )


def _settings_payload_from_args(args: Any) -> Dict[str, Any]:
    """Persist every UI/configurable analysis value except secrets."""
    out: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if key == "api_key":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, list):
            out[key] = list(value)
        elif isinstance(value, dict):
            out[key] = dict(value)
        else:
            out[key] = str(value)
    return out


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

    start_label = "Prä-Analyse & Einstellungen starten" if bool(getattr(args, "tuning_only", False)) else (("Explainer-Pass starten" if bool(getattr(args, "enable_explainer_pass", False)) and not bool(getattr(args, "enable_review_pass", False)) and not bool(getattr(args, "enable_reconstruction_pass", False)) else "Postprocessing starten") if bool(getattr(args, "postprocess_only", False)) else "Analyse starten")
    start_button = st.button(start_label, type="primary", use_container_width=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics = st.empty()
    current_step_text = st.empty()
    event_log = st.empty()
    init_events: List[str] = []

    def show_live_step(stage: str, message: str, *, progress: float = 0.0, detail: str = "") -> None:
        progress_bar.progress(min(1.0, max(0.0, float(progress))))
        status_text.markdown(f"**[{stage}]** {message}")
        cols = metrics.columns(4)
        cols[0].metric("Verarbeitet", "0/—")
        cols[1].metric("Abgeschlossen", "0")
        cols[2].metric("Übersprungen", "0")
        cols[3].metric("Kosten kumulativ", format_eur(0.0))
        current_step_text.markdown(f"**Aktueller Schritt:** {stage} – {message}")
        suffix = f" — {detail}" if detail else ""
        init_events.append(f"- [{stage}/init] {message}{suffix}")
        if len(init_events) > 20:
            del init_events[0]
        event_log.markdown("**Live-Log (neueste unten)**\n" + "\n".join(init_events))

    if not start_button:
        st.info(f"Setze deine Einstellungen und klicke auf **{start_label}**.")
        return

    if not args.api_key:
        st.error("Bitte gib einen API Key ein.")
        return

    env_name = "OPENAI_API_KEY" if args.llm_provider == "openai" else "GEMINI_API_KEY"
    os.environ[env_name] = args.api_key

    try:
        show_live_step("initialisierung", "Lade Topic-Tree …", progress=0.03, detail=args.topics)
        topic_tree = load_json(args.topics)
        show_live_step("initialisierung", "Baue Topic-Katalog …", progress=0.07)
        catalog, key_map = build_topic_catalog(topic_tree)
        topic_keys = [row["topicKey"] for row in catalog]
        topic_catalog_text = format_topic_catalog_for_prompt(catalog)

        show_live_step("initialisierung", f"Erzeuge JSON-Schemas für {len(topic_keys)} Topic-Keys …", progress=0.11)
        schema_a = schema_pass_a(topic_keys)
        schema_b = schema_pass_b(topic_keys)
        schema_review = schema_review_pass(topic_keys)
        schema_reconstruction = schema_reconstruction_pass()
        schema_explainer = schema_explainer_pass()
        schema_cluster_refinement = schema_abstraction_cluster_refinement()

        show_live_step("initialisierung", "Lade Input-Datensatz …", progress=0.16, detail=args.input)
        data = load_json(args.input)
        if isinstance(data, dict) and "questions" in data:
            questions = data["questions"]
            container: Optional[Dict[str, Any]] = data
        elif isinstance(data, list):
            questions = data
            container = None
        else:
            raise ValueError("Input muss Liste oder Objekt mit 'questions' sein.")

        show_live_step("initialisierung", f"Datensatz geladen ({len(questions)} Fragen).", progress=0.22)
        if args.cleanup_spec:
            show_live_step("initialisierung", "Lade Cleanup-Spezifikation …", progress=0.25, detail=args.cleanup_spec)
        cleanup_spec = load_json(args.cleanup_spec) if args.cleanup_spec else None
        if args.images_zip:
            show_live_step("initialisierung", "Bereite Fragenbilder vor …", progress=0.30, detail=args.images_zip)
        image_store = _prepare_image_store(args)
        if args.knowledge_zip or args.knowledge_index:
            show_live_step("knowledge", "Lade/baue Knowledge Base …", progress=0.36, detail=args.knowledge_index or args.knowledge_zip)
        else:
            show_live_step("knowledge", "Knowledge Base deaktiviert/übersprungen.", progress=0.36)
        knowledge_base = _prepare_knowledge_base(args, topic_tree)
        kb_chunks = len(getattr(knowledge_base, "chunks", []) or []) if knowledge_base is not None else 0
        show_live_step("initialisierung", f"Initialisierung abgeschlossen (Knowledge-Chunks: {kb_chunks}).", progress=0.42)

        auto_report: Optional[str] = None
        if bool(getattr(args, "tuning_only", False)) and not bool(getattr(args, "postprocess_only", False)):
            show_live_step("autotune", "Analysiere Datensatzprofil, Retrieval-Qualität und Beispiel-Fragen …", progress=0.50)
            current_settings = _settings_payload_from_args(args)
            show_live_step("autotune", "Frage Modell nach robusten Parametern und Kostenabschätzung …", progress=0.62, detail=args.passB_model or args.passA_model)
            recommendations, auto_report, cost_estimate = recommend_settings(
                provider=args.llm_provider,
                api_key=args.api_key,
                model=args.passB_model or args.passA_model,
                topic_tree=topic_tree,
                questions=questions,
                current=current_settings,
                knowledge_base=knowledge_base,
                models={
                    "passA": args.passA_model,
                    "passB": args.passB_model,
                    "review": args.review_model,
                    "reconstruction": args.reconstruction_model,
                    "explainer": args.explainer_model,
                },
            )
            show_live_step("autotune", f"Empfehlungen erhalten ({len(recommendations)} Parameter).", progress=0.82)
            for key, value in recommendations.items():
                if hasattr(args, key):
                    setattr(args, key, value)

        if bool(getattr(args, "tuning_only", False)):
            if not auto_report:
                auto_report = "Keine Auto-Tuning-Ergebnisse verfügbar."
            tuning_payload = {
                "llm_provider": args.llm_provider,
                "created_from_input": args.input,
                "settings": _settings_payload_from_args(args),
                "report": auto_report,
                "cost_estimate": cost_estimate,
            }
            target_cfg = args.save_tuning_config_path or _resolve_path(folder=os.path.dirname(args.input), filename="analysis_config.json")
            show_live_step("autotune", "Speichere Parameter-Konfiguration …", progress=0.92, detail=target_cfg)
            from ai_exam_analyzer.io_utils import save_json
            save_json(target_cfg, tuning_payload)
            show_live_step("autotune", "Prä-Analyse & Einstellungen abgeschlossen.", progress=1.0, detail=target_cfg)
            st.success(f"Prä-Analyse & Einstellungen abgeschlossen. Konfig gespeichert: {target_cfg}")
            st.info("**Auto-Konfig Bericht**\n\n" + auto_report)
            cost_total = (cost_estimate.get("total") or {})
            st.metric("Geschätzte Gesamtkosten", cost_total.get("costFormatted") or format_eur(float(cost_total.get("costEur") or 0.0)))
            st.metric("Standardabweichung der Schätzung", cost_total.get("standardDeviationFormatted") or format_eur(float(cost_total.get("standardDeviationEur") or 0.0)))
            tuning_cost = (cost_estimate.get("tuningRequest") or {})
            st.metric("Kosten dieser Parameter-Abfrage", tuning_cost.get("costFormatted") or format_eur(float(tuning_cost.get("costEur") or 0.0)))
            profile_rows = []
            for profile_name, payload in (cost_estimate.get("profileEstimates") or {}).items():
                total_payload = payload.get("total") or {}
                by_stage = ((payload.get("estimate") or {}).get("byStage") or {})
                profile_rows.append({
                    "Voreinstellung": payload.get("label") or profile_name,
                    "Profil-Key": profile_name,
                    "Gesamt": total_payload.get("costFormatted") or format_eur(float(total_payload.get("costEur") or 0.0)),
                    "Stdabw.": total_payload.get("standardDeviationFormatted") or format_eur(float(total_payload.get("standardDeviationEur") or 0.0)),
                    "Initialisierung": (by_stage.get("initialization_and_loading") or {}).get("costFormatted") or format_eur(0.0),
                    "Retrieval/Kontext": (by_stage.get("retrieval_and_context_building") or {}).get("costFormatted") or format_eur(0.0),
                    "Basis A": (by_stage.get("pass_a") or by_stage.get("base_pass_a") or {}).get("costFormatted") or format_eur(0.0),
                    "Pass B geschätzt": (by_stage.get("pass_b") or by_stage.get("base_pass_b_estimated") or {}).get("costFormatted") or format_eur(0.0),
                    "Review/Pass C": (by_stage.get("review") or by_stage.get("review_pass_estimated") or {}).get("costFormatted") or format_eur(0.0),
                    "Cluster-Refinement": (by_stage.get("abstraction_cluster_refinement") or {}).get("costFormatted") or format_eur(0.0),
                    "Repeat-Reconstruction": (by_stage.get("repeat_reconstruction") or {}).get("costFormatted") or format_eur(0.0),
                    "Reconstruction": (by_stage.get("reconstruction") or by_stage.get("reconstruction_pass") or {}).get("costFormatted") or format_eur(0.0),
                    "Explainer": (by_stage.get("explainer") or by_stage.get("explainer_pass") or {}).get("costFormatted") or format_eur(0.0),
                    "Output/Report": (by_stage.get("output_and_cost_report") or {}).get("costFormatted") or format_eur(0.0),
                })
            if profile_rows:
                st.subheader("Kostenvergleich aller Voreinstellungen")
                st.caption("Repeat-Reconstruction ist ein deterministischer Abgleich wiederholter Frage-/Antwortmuster über Cluster/Jahrgänge; deshalb entstehen dafür keine LLM-Tokens, der Pass wird aber bewusst in der Aufschlüsselung gezeigt.")
                st.dataframe(profile_rows, use_container_width=True, hide_index=True)
            st.json(cost_estimate)
            return

        show_live_step("pipeline", "Starte Analyse-Workflow …", progress=0.45)
        recent_events: List[str] = list(init_events[-8:])
        latest_cost_total_formatted = format_eur(0.0)

        def on_progress(event: Dict[str, Any]) -> None:
            nonlocal latest_cost_total_formatted
            total = max(1, int(event.get("total") or len(questions) or 1))
            processed = int(event.get("processed", 0) or 0)
            done_count = int(event.get("done", 0) or 0)
            skipped_count = int(event.get("skipped", 0) or 0)
            index = event.get("index")
            stage = str(event.get("stage") or "pipeline")
            event_name = str(event.get("event") or "event")
            message = str(event.get("message") or "---")

            pct = min(1.0, processed / total)
            progress_bar.progress(pct)

            headline = message
            if index is not None:
                headline = f"{headline} *(Frage {index}/{total})*"
            status_text.markdown(f"**[{stage}]** {headline}")

            if "cost_total_formatted" in event or "cost_total_eur" in event:
                latest_cost_total_formatted = str(event.get("cost_total_formatted") or format_eur(float(event.get("cost_total_eur") or 0.0)))

            cols = metrics.columns(4)
            cols[0].metric("Verarbeitet", f"{processed}/{total}")
            cols[1].metric("Abgeschlossen", str(done_count))
            cols[2].metric("Übersprungen", str(skipped_count))
            cols[3].metric("Kosten kumulativ", latest_cost_total_formatted)
            current_step_text.markdown(f"**Aktueller Schritt:** {stage} – {headline}")

            details = []
            if "retrieval_quality" in event:
                details.append(f"rq={float(event.get('retrieval_quality') or 0.0):.2f}")
            if "evidence_count" in event:
                details.append(f"evidence={int(event.get('evidence_count') or 0)}")
            if "cost_stage_eur" in event or "cost_stage_formatted" in event:
                details.append(f"cost_step={event.get('cost_stage_formatted') or format_eur(float(event.get('cost_stage_eur') or 0.0))}")
            if "cost_total_eur" in event or "cost_total_formatted" in event:
                details.append(f"cost_total={event.get('cost_total_formatted') or format_eur(float(event.get('cost_total_eur') or 0.0))}")
            detail_text = f" ({', '.join(details)})" if details else ""

            line = f"- [{stage}/{event_name}] {message}{detail_text}"
            recent_events.append(line)
            if len(recent_events) > 20:
                del recent_events[0]
            event_log.markdown("**Live-Log (neueste unten)**\n" + "\n".join(recent_events))

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
            schema_cluster_refinement=schema_cluster_refinement,
            cleanup_spec=cleanup_spec,
            knowledge_base=knowledge_base,
            image_store=image_store,
            progress_callback=on_progress,
        )

        progress_bar.progress(1.0)
        st.success(f"Analyse beendet. Ergebnis gespeichert unter: {args.output}")
        st.caption("Kosten-/Token-Report: automatisch neben der Ausgabe als `.costs.json` gespeichert (oder über COST_REPORT_PATH konfigurierbar).")
        if auto_report:
            st.info("**Auto-Konfig Bericht**\n\n" + auto_report)

    except Exception as exc:
        st.exception(exc)


if __name__ == "__main__":
    main()
