#!/usr/bin/env python3
"""DocsDocs topic annotation + answer plausibility auditing (2-pass).

Merged from the earlier 'classify_topics.py' pipeline (enum-safe topicKey + rich aiAudit structure)
and the newer V2 wrapper (CLI + robust handling for reasoning models that reject temperature).

Pass A (Proposer): fast model (default: gpt-4o-mini)
  1) topic_initial classification (text+answers) with confidence
  2) answer plausibility check; may propose correction with confidence
  3) maintenance flagging
  4) topic_final re-check after analysis

Pass B (Verifier): reasoning model (default: o4-mini)
  Runs only for risky cases (configurable triggers) and confirms answer changes + may override topic.

Output:
  - Writes aiAudit per question.
  - Optional convenience top-level fields prefixed with 'ai*' to avoid collisions.

Example:
  python classify_topics_merged.py --input dxport.annotated.json --topics topic-tree.json --output dxport.annotated.fixed.json --resume --write-top-level
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI



# -----------------------------
# CONFIG (edit these defaults to avoid typing CLI options each run)
# You can still override any of these via command-line flags.
# Paths can be relative to the working directory you run the script from.
# -----------------------------
CONFIG = {
    "INPUT_PATH": "export.json",
    "TOPICS_PATH": "topic-tree.json",
    "OUTPUT_PATH": "export.AIannotated.json",

    # Execution
    "RESUME": False,              # skip already completed questions (same pipelineVersion)
    "LIMIT": 0,                   # 0 = all questions
    "CHECKPOINT_EVERY": 10,       # save after N processed questions
    "SLEEP": 0.15,                # seconds between questions

    # Models
    "PASSA_MODEL": "gpt-4o-mini",
    "PASSB_MODEL": "o4-mini",
    "PASSA_TEMPERATURE": 0.0,     # ignored automatically for reasoning models
    "PASSB_REASONING_EFFORT": "high",  # low|medium|high

    # Pass-B triggers
    "TRIGGER_ANSWER_CONF": 0.80,
    "TRIGGER_TOPIC_CONF": 0.85,

    # Apply answer changes only if verifier confidence >= this
    "APPLY_CHANGE_MIN_CONF_B": 0.80,

    # Output extras
    "WRITE_TOP_LEVEL": True,      # add aiSuperTopic/aiSubtopic/... fields on each question
    "DEBUG": False,               # store raw pass outputs under aiAudit._debug (bigger files)
}

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def is_reasoning_model(model: str) -> bool:
    """Heuristic: o-series + gpt-5* are treated as reasoning models (may reject temperature/top_p)."""
    m = (model or "").lower().strip()
    return m.startswith("o") or m.startswith("gpt-5")


def normalize_indices(indices: List[int], n_answers: int) -> List[int]:
    return sorted({i for i in indices if isinstance(i, int) and 0 <= i < n_answers})


def apply_correct_indices(q: Dict[str, Any], new_indices: List[int]) -> None:
    """Update correctIndices + answers[].isCorrect + correctAnswers."""
    answers = q.get("answers") or []
    new_set = set(new_indices)

    for i, a in enumerate(answers):
        a["isCorrect"] = (i in new_set)

    q["correctIndices"] = list(new_indices)
    q["correctAnswers"] = [
        {"index": i, "text": answers[i].get("text"), "html": answers[i].get("html")}
        for i in new_indices
        if 0 <= i < len(answers)
    ]


# -----------------------------
# Topic catalog (topicKey enum + text mapping)
# -----------------------------
def build_topic_catalog(topic_tree: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Build deterministic topic catalog and a topicKey->row map."""
    catalog: List[Dict[str, Any]] = []
    key_map: Dict[str, Dict[str, Any]] = {}

    super_topics = topic_tree.get("superTopics", [])
    if not isinstance(super_topics, list) or not super_topics:
        raise ValueError("Topic tree must have non-empty 'superTopics' array.")

    for s_idx, st in enumerate(super_topics, start=1):
        super_name = (st.get("name") or "").strip()
        subs = st.get("subtopics") or []
        if not isinstance(subs, list) or not subs:
            raise ValueError(f"SuperTopic '{super_name}' must have non-empty 'subtopics' list.")
        for sub_idx, sub_name in enumerate(subs, start=1):
            sub_name = (sub_name or "").strip()
            topic_key = f"{s_idx}:{sub_idx}"
            row = {
                "superTopicId": s_idx,
                "superTopicName": super_name,
                "subtopicId": sub_idx,
                "subtopicName": sub_name,
                "topicKey": topic_key,
            }
            catalog.append(row)
            key_map[topic_key] = row

    return catalog, key_map


def format_topic_catalog_for_prompt(catalog: List[Dict[str, Any]]) -> str:
    lines = ["Erlaubte Topics (wähle genau EINEN topicKey):"]
    current_super = None
    for row in catalog:
        if row["superTopicId"] != current_super:
            current_super = row["superTopicId"]
            lines.append(f'\n{row["superTopicId"]}. {row["superTopicName"]}')
        lines.append(f'  - topicKey {row["topicKey"]}: {row["subtopicName"]}')
    return "\n".join(lines)


# -----------------------------
# Prompt payload
# -----------------------------
def build_question_payload(q: Dict[str, Any]) -> Dict[str, Any]:
    answers = q.get("answers") or []
    answer_texts = []
    for i, a in enumerate(answers):
        answer_texts.append({
            "index": i,
            "id": a.get("id"),
            "text": (a.get("text") or "").strip()
        })

    return {
        "questionId": q.get("id"),
        "questionText": (q.get("questionText") or "").strip(),
        "answers": answer_texts,
        "currentCorrectIndices": q.get("correctIndices") or [],
        "explanationText": (q.get("explanationText") or "").strip(),
        "hasImages": bool(q.get("imageUrls") or q.get("imageFiles")),
    }


# -----------------------------
# Schemas (Structured Outputs)
# -----------------------------
def schema_pass_a(topic_keys: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "topic_initial": {
                "type": "object",
                "properties": {
                    "topicKey": {"type": "string", "enum": topic_keys},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reasonShort": {"type": "string"},
                },
                "required": ["topicKey", "confidence", "reasonShort"],
                "additionalProperties": False,
            },
            "answer_review": {
                "type": "object",
                "properties": {
                    "isPlausible": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "recommendChange": {"type": "boolean"},
                    "proposedCorrectIndices": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "minItems": 0,
                    },
                    "reasonShort": {"type": "string"},
                    "maintenanceSuspicion": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "isPlausible",
                    "confidence",
                    "recommendChange",
                    "proposedCorrectIndices",
                    "reasonShort",
                    "maintenanceSuspicion",
                ],
                "additionalProperties": False,
            },
            "maintenance": {
                "type": "object",
                "properties": {
                    "needsMaintenance": {"type": "boolean"},
                    "severity": {"type": "integer", "minimum": 1, "maximum": 3},
                    "reasons": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["needsMaintenance", "severity", "reasons"],
                "additionalProperties": False,
            },
            "topic_final": {
                "type": "object",
                "properties": {
                    "topicKey": {"type": "string", "enum": topic_keys},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reasonShort": {"type": "string"},
                },
                "required": ["topicKey", "confidence", "reasonShort"],
                "additionalProperties": False,
            },
        },
        "required": ["topic_initial", "answer_review", "maintenance", "topic_final"],
        "additionalProperties": False,
    }


def schema_pass_b(topic_keys: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "verify_answer": {
                "type": "object",
                "properties": {
                    "agreeWithChange": {"type": "boolean"},
                    "verifiedCorrectIndices": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "minItems": 0,
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reasonShort": {"type": "string"},
                    "cannotJudge": {"type": "boolean"},
                },
                "required": ["agreeWithChange", "verifiedCorrectIndices", "confidence", "reasonShort", "cannotJudge"],
                "additionalProperties": False,
            },
            "maintenance": {
                "type": "object",
                "properties": {
                    "needsMaintenance": {"type": "boolean"},
                    "severity": {"type": "integer", "minimum": 1, "maximum": 3},
                    "reasons": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["needsMaintenance", "severity", "reasons"],
                "additionalProperties": False,
            },
            "topic_final": {
                "type": "object",
                "properties": {
                    "topicKey": {"type": "string", "enum": topic_keys},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reasonShort": {"type": "string"},
                },
                "required": ["topicKey", "confidence", "reasonShort"],
                "additionalProperties": False,
            },
        },
        "required": ["verify_answer", "maintenance", "topic_final"],
        "additionalProperties": False,
    }


# -----------------------------
# OpenAI call wrapper
# -----------------------------
def call_json_schema(
    client: OpenAI,
    *,
    model: str,
    system: str,
    user: str,
    schema: Dict[str, Any],
    format_name: str,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    max_output_tokens: int = 900,
) -> Dict[str, Any]:
    """Responses API + Structured Outputs (json_schema). Handles temperature for reasoning models."""

    def _do_call(send_temperature: bool) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": format_name,
                    "schema": schema,
                    "strict": True,
                }
            },
            "max_output_tokens": max_output_tokens,
        }

        if send_temperature and temperature is not None:
            params["temperature"] = temperature

        if is_reasoning_model(model) and reasoning_effort:
            params["reasoning"] = {"effort": reasoning_effort}

        resp = client.responses.create(**params)
        if resp.status != "completed":
            raise RuntimeError(f"Response not completed: {resp.status}")
        return json.loads(resp.output_text)

    # reasoning: never send temperature
    if is_reasoning_model(model):
        return _do_call(send_temperature=False)

    # non-reasoning: try with temperature; if server rejects, retry without
    try:
        return _do_call(send_temperature=True)
    except Exception as e:
        msg = str(e)
        if "temperature" in msg and ("Unsupported parameter" in msg or "not supported" in msg):
            return _do_call(send_temperature=False)
        raise


# -----------------------------
# Pass runners
# -----------------------------
def run_pass_a(
    client: OpenAI,
    *,
    topic_catalog_text: str,
    payload: Dict[str, Any],
    schema: Dict[str, Any],
    model: str,
    temperature: float,
) -> Dict[str, Any]:
    system = (
        "Du bist ein strenger Prüfungsfragen-Analyst und Klassifikator.\n"
        "Arbeitsablauf:\n"
        "1) Ordne die Frage zunächst NUR anhand von Fragetext + Antwortoptionen einem Topic (topicKey) zu.\n"
        "2) Prüfe, ob die aktuell markierte(n) richtige(n) Antwort(en) inhaltlich plausibel sind.\n"
        "   - Wenn eindeutig falsch, schlage korrigierte CorrectIndices vor.\n"
        "   - Bei Unsicherheit: recommendChange=false und confidence niedrig.\n"
        "3) Markiere Wartungsbedarf (unklar, mehrdeutig, widersprüchlich, Bild nötig, etc.).\n"
        "4) Ordne nach der inhaltlichen Analyse das Topic ggf. neu zu.\n\n"
        "Regeln:\n"
        "- Antworte ausschließlich im vorgegebenen JSON-Schema.\n"
        "- proposedCorrectIndices sind 0-basiert.\n\n"
        f"{topic_catalog_text}"
    )
    user = json.dumps(payload, ensure_ascii=False)
    return call_json_schema(
        client,
        model=model,
        system=system,
        user=user,
        schema=schema,
        format_name="pass_a_audit",
        temperature=temperature,
        max_output_tokens=1100,
    )


def run_pass_b(
    client: OpenAI,
    *,
    topic_catalog_text: str,
    payload: Dict[str, Any],
    pass_a: Dict[str, Any],
    schema: Dict[str, Any],
    model: str,
    reasoning_effort: str,
) -> Dict[str, Any]:
    system = (
        "Du bist ein unabhängiger Verifier.\n"
        "Du bekommst eine Prüfungsfrage + einen Vorschlag (Pass A).\n"
        "Aufgaben:\n"
        "A) Prüfe, ob die in Pass A empfohlene Änderung der CorrectIndices fachlich korrekt ist.\n"
        "   - agreeWithChange=true nur wenn du die Änderung klar unterstützt.\n"
        "   - cannotJudge=true wenn Bild/Infos fehlen oder die Frage unentscheidbar ist.\n"
        "   - Liefere verifiedCorrectIndices (0-basiert). Wenn cannotJudge=true, gib [] aus.\n"
        "B) Markiere Wartungsbedarf.\n"
        "C) Gib deinen eigenen finalen TopicKey nach deiner Analyse aus.\n\n"
        "Regeln:\n"
        "- Sei konservativ: bei Zweifel keine Änderung bestätigen.\n"
        "- Antworte ausschließlich im vorgegebenen JSON-Schema.\n\n"
        f"{topic_catalog_text}"
    )
    packed = {"question": payload, "passA": pass_a}
    user = json.dumps(packed, ensure_ascii=False)
    return call_json_schema(
        client,
        model=model,
        system=system,
        user=user,
        schema=schema,
        format_name="pass_b_verify",
        temperature=None,  # MUST NOT send temperature to reasoning models
        reasoning_effort=reasoning_effort,
        max_output_tokens=900,
    )


def should_run_pass_b(pass_a: Dict[str, Any], trigger_answer_conf: float, trigger_topic_conf: float) -> bool:
    ar = pass_a["answer_review"]
    m = pass_a["maintenance"]
    t1 = pass_a["topic_initial"]
    tf = pass_a["topic_final"]

    if ar.get("recommendChange") is True:
        return True
    if float(ar.get("confidence", 0.0)) < trigger_answer_conf:
        return True
    if bool(m.get("needsMaintenance")):
        return True
    if float(t1.get("confidence", 0.0)) < trigger_topic_conf:
        return True
    if float(tf.get("confidence", 0.0)) < trigger_topic_conf:
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()

    # Paths (defaults from CONFIG; override via CLI if needed)
    ap.add_argument("--input", default=CONFIG["INPUT_PATH"], help="Input dataset JSON (original or in-progress annotated)")
    ap.add_argument("--topics", default=CONFIG["TOPICS_PATH"], help="Topic tree JSON (superTopics/subtopics)")
    ap.add_argument("--output", default=CONFIG["OUTPUT_PATH"], help="Output JSON path")

    # Execution
    ap.add_argument("--resume", dest="resume", action="store_true", default=CONFIG["RESUME"],
                    help="Skip questions already completed with this pipelineVersion")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="Do not skip completed questions")
    ap.add_argument("--limit", type=int, default=CONFIG["LIMIT"], help="Only process first N questions (0 = all)")
    ap.add_argument("--checkpoint-every", type=int, default=CONFIG["CHECKPOINT_EVERY"],
                    help="Save after every N processed questions")
    ap.add_argument("--sleep", type=float, default=CONFIG["SLEEP"], help="Sleep seconds between questions")

    # Models
    ap.add_argument("--passA-model", default=CONFIG["PASSA_MODEL"])
    ap.add_argument("--passB-model", default=CONFIG["PASSB_MODEL"])
    ap.add_argument("--passA-temperature", type=float, default=CONFIG["PASSA_TEMPERATURE"])
    ap.add_argument("--passB-reasoning-effort", default=CONFIG["PASSB_REASONING_EFFORT"],
                    choices=["low", "medium", "high"])

    # Pass-B triggers
    ap.add_argument("--trigger-answer-conf", type=float, default=CONFIG["TRIGGER_ANSWER_CONF"])
    ap.add_argument("--trigger-topic-conf", type=float, default=CONFIG["TRIGGER_TOPIC_CONF"])

    # Apply answer changes only if verifier confidence >= this
    ap.add_argument("--apply-change-min-conf-b", type=float, default=CONFIG["APPLY_CHANGE_MIN_CONF_B"])

    # Output extras
    ap.add_argument("--write-top-level", dest="write_top_level", action="store_true",
                    default=CONFIG["WRITE_TOP_LEVEL"],
                    help="Also write ai* convenience fields on question level")
    ap.add_argument("--no-write-top-level", dest="write_top_level", action="store_false",
                    help="Do not write ai* convenience fields")

    ap.add_argument("--debug", dest="debug", action="store_true", default=CONFIG["DEBUG"],
                    help="Store raw pass outputs under aiAudit._debug")
    ap.add_argument("--no-debug", dest="debug", action="store_false",
                    help="Do not store raw pass outputs")

    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    topic_tree = load_json(args.topics)
    catalog, key_map = build_topic_catalog(topic_tree)
    topic_keys = [row["topicKey"] for row in catalog]
    topic_catalog_text = format_topic_catalog_for_prompt(catalog)

    schemaA = schema_pass_a(topic_keys)
    schemaB = schema_pass_b(topic_keys)

    data = load_json(args.input)

    if isinstance(data, dict) and "questions" in data:
        questions = data["questions"]
        container = data
    elif isinstance(data, list):
        questions = data
        container = None
    else:
        raise ValueError("Input must be a list of questions or {questions:[...]} object.")

    client = OpenAI()

    pipeline_version = "2pass-merged-v1"
    done = 0
    skipped = 0
    processed = 0

    for i, q in enumerate(questions, start=1):
        if args.limit and processed >= args.limit:
            break

        if args.resume and isinstance(q.get("aiAudit"), dict):
            if q["aiAudit"].get("pipelineVersion") == pipeline_version and q["aiAudit"].get("status") == "completed":
                skipped += 1
                continue

        payload = build_question_payload(q)
        answers = q.get("answers") or []
        n_answers = len(answers)
        current = normalize_indices(q.get("correctIndices") or [], n_answers)

        audit: Dict[str, Any] = {
            "pipelineVersion": pipeline_version,
            "status": "error",
            "models": {"passA": args.passA_model, "passB": None},
        }

        try:
            pass_a = run_pass_a(
                client,
                topic_catalog_text=topic_catalog_text,
                payload=payload,
                schema=schemaA,
                model=args.passA_model,
                temperature=args.passA_temperature,
            )

            proposed = normalize_indices(pass_a["answer_review"]["proposedCorrectIndices"], n_answers)

            # defaults from pass A
            final_topic_key = pass_a["topic_final"]["topicKey"]
            final_topic_conf = float(pass_a["topic_final"]["confidence"])
            final_topic_reason = pass_a["topic_final"]["reasonShort"]
            final_topic_source = "passA"

            maintenance = pass_a["maintenance"]
            extra_flags = pass_a["answer_review"].get("maintenanceSuspicion", []) or []
            merged_reasons = list(dict.fromkeys((maintenance.get("reasons") or []) + extra_flags))

            recommend_a = bool(pass_a["answer_review"]["recommendChange"])
            conf_a = float(pass_a["answer_review"]["confidence"])

            will_change = False
            change_source = "none"
            final_correct_indices = current
            verification: Dict[str, Any] = {"ran": False}

            ran_b = should_run_pass_b(pass_a, args.trigger_answer_conf, args.trigger_topic_conf)
            pass_b: Optional[Dict[str, Any]] = None

            if ran_b:
                try:
                    pass_b = run_pass_b(
                        client,
                        topic_catalog_text=topic_catalog_text,
                        payload=payload,
                        pass_a=pass_a,
                        schema=schemaB,
                        model=args.passB_model,
                        reasoning_effort=args.passB_reasoning_effort,
                    )
                    audit["models"]["passB"] = args.passB_model

                    # merge maintenance with verifier
                    m_b = pass_b["maintenance"]
                    merged_reasons = list(dict.fromkeys(merged_reasons + (m_b.get("reasons") or [])))
                    maintenance = {
                        "needsMaintenance": bool(maintenance.get("needsMaintenance")) or bool(m_b.get("needsMaintenance")),
                        "severity": int(max(int(maintenance.get("severity", 1)), int(m_b.get("severity", 1)))),
                        "reasons": merged_reasons,
                    }

                    # topic final from verifier
                    final_topic_key = pass_b["topic_final"]["topicKey"]
                    final_topic_conf = float(pass_b["topic_final"]["confidence"])
                    final_topic_reason = pass_b["topic_final"]["reasonShort"]
                    final_topic_source = "passB"

                    # answer verification
                    v = pass_b["verify_answer"]
                    cannot = bool(v.get("cannotJudge"))
                    agree = bool(v.get("agreeWithChange"))
                    conf_b = float(v.get("confidence"))
                    verified = normalize_indices(v.get("verifiedCorrectIndices", []), n_answers)

                    if (not cannot) and agree and (conf_b >= args.apply_change_min_conf_b) and len(verified) > 0 and verified != current:
                        apply_correct_indices(q, verified)
                        will_change = True
                        change_source = "passB"
                        final_correct_indices = verified
                    else:
                        final_correct_indices = current

                    verification = {
                        "ran": True,
                        "model": args.passB_model,
                        "cannotJudge": cannot,
                        "agreeWithChange": agree,
                        "confidence": conf_b,
                        "verifiedCorrectIndices": verified,
                        "appliedChange": will_change,
                    }

                except Exception as e:
                    audit["models"]["passB"] = args.passB_model
                    verification = {"ran": True, "model": args.passB_model, "error": str(e)}
                    maintenance = {
                        "needsMaintenance": bool(maintenance.get("needsMaintenance")),
                        "severity": int(maintenance.get("severity", 1)),
                        "reasons": merged_reasons,
                    }

            else:
                maintenance = {
                    "needsMaintenance": bool(maintenance.get("needsMaintenance")),
                    "severity": int(maintenance.get("severity", 1)),
                    "reasons": merged_reasons,
                }

            init_row = key_map[pass_a["topic_initial"]["topicKey"]]
            final_row = key_map[final_topic_key]

            audit.update({
                "status": "completed",
                "topicInitial": {
                    "superTopic": init_row["superTopicName"],
                    "subtopic": init_row["subtopicName"],
                    "confidence": float(pass_a["topic_initial"]["confidence"]),
                    "reasonShort": pass_a["topic_initial"]["reasonShort"],
                },
                "topicFinal": {
                    "superTopic": final_row["superTopicName"],
                    "subtopic": final_row["subtopicName"],
                    "confidence": final_topic_conf,
                    "reasonShort": final_topic_reason,
                    "source": final_topic_source,
                },
                "answerPlausibility": {
                    "originalCorrectIndices": current,
                    "passA": {
                        "isPlausible": bool(pass_a["answer_review"]["isPlausible"]),
                        "confidence": conf_a,
                        "recommendChange": recommend_a,
                        "proposedCorrectIndices": proposed,
                        "reasonShort": pass_a["answer_review"]["reasonShort"],
                    },
                    "finalCorrectIndices": final_correct_indices,
                    "changedInDataset": bool(will_change),
                    "changeSource": change_source,
                    "verification": verification,
                },
                "maintenance": maintenance,
            })

            if args.debug:
                audit["_debug"] = {"passA_raw": pass_a, "passB_raw": pass_b}

            if args.write_top_level:
                q["aiSuperTopic"] = audit["topicFinal"]["superTopic"]
                q["aiSubtopic"] = audit["topicFinal"]["subtopic"]
                q["aiTopicConfidence"] = audit["topicFinal"]["confidence"]
                q["aiNeedsMaintenance"] = audit["maintenance"]["needsMaintenance"]
                q["aiMaintenanceSeverity"] = audit["maintenance"]["severity"]
                q["aiMaintenanceReasons"] = audit["maintenance"]["reasons"]

            done += 1

        except Exception as e:
            audit["status"] = "error"
            audit["error"] = str(e)

        q["aiAudit"] = audit

        processed += 1
        if args.checkpoint_every and processed % args.checkpoint_every == 0:
            out_obj = container if container is not None else questions
            save_json(args.output, out_obj)
            print(f"[{i}/{len(questions)}] checkpoint | processed={processed} done={done} skipped={skipped} lastStatus={audit.get('status')}")

        time.sleep(args.sleep)

    out_obj = container if container is not None else questions
    save_json(args.output, out_obj)
    print(f"Finished. processed={processed} done={done} skipped={skipped}. Output: {args.output}")


if __name__ == "__main__":
    main()
