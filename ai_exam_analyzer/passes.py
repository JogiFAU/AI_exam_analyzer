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
        max_output_tokens=2200,
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
        max_output_tokens=2200,
        max_retries=4,
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



def run_reconstruction_pass(
    client: Any,
    *,
    payload: Dict[str, Any],
    schema: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    system = (
        "Du bist ein Senior-Redakteur für universitäre Prüfungsaufgaben.\n"
        "Analysiere eine Frage im Kontext von Cluster- und Wissenshinweisen.\n"
        "Ziele:\n"
        "1) Beurteile, ob die Frage vermutlich eine unvollständig übernommene Altfrage ist.\n"
        "2) Erstelle eine rekonstruierte Version in gleicher Struktur (Fragetext + Antworten).\n"
        "3) Bei qualitativ hochwertigen Fragen nur sprachlich/präziser verbessern, nicht inhaltlich verändern.\n"
        "4) Bei schwachen Fragen fehlende/unklare Teile evidenzbasiert ergänzen (Altfragen-Hinweise vor KB).\n"
        "Spezialregel für Antwortoptionen:\n"
        "- Wenn die Frage weniger als 4 Antwortmöglichkeiten hat, versuche fehlende Optionen aus ähnlichen Cluster-Fragen zu ergänzen, falls inhaltlich plausibel.\n"
        "- Wenn das nicht möglich ist, ergänze fehlende Optionen anhand relevanter Fakten aus retrievedEvidence (Knowledge-Base).\n"
        "- Wenn beides nicht belastbar möglich ist: recommendManualReview=true und reconstructionStrategy='no_completion_manual_review'.\n"
        "- Keine Mehrfachauswahl einführen: Es bleibt bei genau einer korrekten Antwortoption.\n"
        "- Wenn bereits eine korrekte Antwort vorhanden ist, keine zweite korrekte Antwort hinzufügen.\n"
        "Hinweis: Das Stichwort 'Altfrage' ist ein starkes Legacy-Signal.\n"
        "Antworte strikt im JSON-Schema."
    )
    user = [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}]
    return call_json_schema(
        client,
        model=model,
        system=system,
        user=user,
        schema=schema,
        format_name="reconstruction_pass",
        temperature=0.0,
        max_output_tokens=3000,
        max_retries=4,
    )


def run_explainer_pass(
    client: Any,
    *,
    payload: Dict[str, Any],
    schema: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    system = (
        "Du bist ein didaktisch starker Fach-Tutor.\n"
        "Erkläre ausführlich die korrekte Lösung auf Basis von Frage + Kontext (Evidenz, Cluster, etc.).\n"
        "Erkläre außerdem, warum die falschen Optionen falsch sind und ordne die Frage fachlich ein.\n"
        "Antworte strikt im JSON-Schema."
    )
    user = [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}]
    return call_json_schema(
        client,
        model=model,
        system=system,
        user=user,
        schema=schema,
        format_name="explainer_pass",
        temperature=0.2,
        max_output_tokens=2200,
    )
