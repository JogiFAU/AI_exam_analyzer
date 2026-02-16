"""Pass A and Pass B runner functions."""

import json
from typing import Any, Dict

from ai_exam_analyzer.openai_client import call_json_schema


def run_pass_a(
    client: Any,
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
        "2) Versuche die Frage nach deinem Wissen zu beantworten. Mehrere Antworten können eventuell richtig sein. \n"
        "3) Vergleiche deine Antwort(en) mit der/den aktuell markierte(n) richtige(n) Antwort(en). Sollte(n) deine Antwort(en) sich unterscheiden, prüfe welche Inhaltlich plausibel sind.\n"
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
    client: Any,
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
        temperature=None,
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
