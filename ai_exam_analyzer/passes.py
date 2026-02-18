"""Pass runner functions."""

import json
from typing import Any, Dict, List

from ai_exam_analyzer.openai_client import call_json_schema


def run_pass_a(
    client: Any,
    *,
    topic_catalog_text: str,
    payload: Dict[str, Any],
    schema: Dict[str, Any],
    model: str,
    temperature: float,
    question_images: List[Dict[str, Any]],
) -> Dict[str, Any]:
    system = (
        "Du bist ein strenger Prüfungsfragen-Analyst und Klassifikator.\n"
        "Arbeitsablauf:\n"
        "1) Nutze topicCandidates als priorisierte Vorauswahl und bilde topic_initial primär innerhalb dieser Kandidaten.\n"
        "   - Nur bei starker fachlicher Evidenz darfst du außerhalb der Kandidaten entscheiden.\n"
        "2) Nutze imageContext, imageClusterContext und knowledgeImageContext für die Bildzuordnung.\n"
        "   - Wenn Bild vorhanden ist: bewerte visuelle Hinweise zwingend mit.\n"
        "   - Falls knowledgeImageContext ähnliche Knowledge-Base-Bilder enthält, nutze deren Kontext aktiv.\n"
        "3) Beantworte die Frage fachlich mit retrievedEvidence + Bildkontext.\n"
        "   - Wenn retrievedEvidence leer oder schwach ist, setze confidence konservativ und markiere Wartungsbedarf.\n"
        "4) Vergleiche mit currentCorrectIndices und entscheide über recommendChange/proposedCorrectIndices.\n"
        "5) Markiere Wartungsbedarf (unklar, mehrdeutig, fehlendes Bild, etc.).\n"
        "6) Bestimme topic_final und gib eine Ein-Satz-Abstraktion question_abstraction.summary aus.\n\n"
        "Regeln:\n"
        "- evidenceChunkIds muss genutzte Chunk-IDs aus retrievedEvidence referenzieren (oder []).\n"
        "- Nutze nur Evidenz, die fachlich direkt zur Frage passt; vermeide spekulative Schlüsse.\n"
        "- proposedCorrectIndices/verifiedCorrectIndices/finalCorrectIndices verwenden answerIndex (1-basiert), nicht Array-Position.\n"
        "- Wenn Bild erwartet wird, aber fehlt: needsMaintenance=true.\n"
        "- Antworte ausschließlich im vorgegebenen JSON-Schema.\n\n"
        f"{topic_catalog_text}"
    )
    user = [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}] + question_images
    return call_json_schema(
        client,
        model=model,
        system=system,
        user=user,
        schema=schema,
        format_name="pass_a_audit",
        temperature=temperature,
        max_output_tokens=1200,
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
    question_images: List[Dict[str, Any]],
) -> Dict[str, Any]:
    system = (
        "Du bist ein unabhängiger Verifier.\n"
        "Du bekommst eine Prüfungsfrage + Pass A Ergebnis.\n"
        "Prüfe mit retrievedEvidence + Bildkontext + Clusterkontext, ob die vorgeschlagene Korrektur fachlich stimmt.\n"
        "Bei schwacher/fehlender Evidenz: konservativ bleiben und cannotJudge erwägen.\n"
        "Berücksichtige Bildähnlichkeits-Hinweise aus knowledgeImageContext zwingend.\n"
        "Wenn Bild fehlt oder Fall unentscheidbar: cannotJudge=true und Wartungsbedarf markieren.\n"
        "Antworte ausschließlich im JSON-Schema.\n\n"
        f"{topic_catalog_text}"
    )
    packed = {"question": payload, "passA": pass_a}
    user = [{"type": "input_text", "text": json.dumps(packed, ensure_ascii=False)}] + question_images
    return call_json_schema(
        client,
        model=model,
        system=system,
        user=user,
        schema=schema,
        format_name="pass_b_verify",
        temperature=None,
        reasoning_effort=reasoning_effort,
        max_output_tokens=1000,
    )


def run_review_pass(
    client: Any,
    *,
    payload: Dict[str, Any],
    current_audit: Dict[str, Any],
    schema: Dict[str, Any],
    model: str,
    question_images: List[Dict[str, Any]],
) -> Dict[str, Any]:
    system = (
        "Du bist ein sehr strenger Senior-Reviewer für wartungsbedürftige Fragen.\n"
        "Nutze den gesamten Kontext (Frage, Wissen, Bildcluster, Knowledge-Bildtreffer, Audit-Historie).\n"
        "Korrigiere ggf. finalCorrectIndices/finalTopicKey und gib reviewComment.\n"
        "Setze recommendManualReview=true, wenn Unsicherheit oder Datenprobleme bestehen.\n"
        "Antworte nur im JSON-Schema."
    )
    packed = {"question": payload, "currentAudit": current_audit}
    user = [{"type": "input_text", "text": json.dumps(packed, ensure_ascii=False)}] + question_images
    return call_json_schema(
        client,
        model=model,
        system=system,
        user=user,
        schema=schema,
        format_name="pass_c_review",
        temperature=0.0,
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
