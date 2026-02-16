"""Prompt payload builders."""

from typing import Any, Dict


def build_question_payload(q: Dict[str, Any]) -> Dict[str, Any]:
    answers = q.get("answers") or []
    answer_texts = []
    for i, a in enumerate(answers):
        answer_texts.append({
            "index": i,
            "id": a.get("id"),
            "text": (a.get("text") or "").strip(),
        })

    image_refs = [str(ref).strip() for ref in (q.get("imageFiles") or []) if str(ref).strip()]
    image_urls = [str(url).strip() for url in (q.get("imageUrls") or []) if str(url).strip()]

    return {
        "questionId": q.get("id"),
        "questionText": (q.get("questionText") or "").strip(),
        "answers": answer_texts,
        "currentCorrectIndices": q.get("correctIndices") or [],
        "explanationText": (q.get("explanationText") or "").strip(),
        "hasImages": bool(image_urls or image_refs),
        "imageRefs": image_refs,
        "imageUrls": image_urls,
    }
