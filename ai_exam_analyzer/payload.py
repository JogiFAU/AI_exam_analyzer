"""Prompt payload builders."""

from typing import Any, Dict, List, Optional


def _derive_answer_index(answer: Dict[str, Any], fallback_position: int) -> int:
    """Return stable external answer index (1-based by default)."""
    for key in ("answerIndex", "position", "index"):
        value = answer.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return fallback_position


def build_question_payload(q: Dict[str, Any], *, current_correct_indices: Optional[List[int]] = None) -> Dict[str, Any]:
    answers = q.get("answers") or []
    answer_texts = []
    for i, a in enumerate(answers):
        answer_index = _derive_answer_index(a, i + 1)
        answer_texts.append({
            "arrayPosition": i,
            "answerIndex": answer_index,
            "id": a.get("id"),
            "text": (a.get("text") or "").strip(),
        })

    image_refs = [str(ref).strip() for ref in (q.get("imageFiles") or []) if str(ref).strip()]
    image_urls = [str(url).strip() for url in (q.get("imageUrls") or []) if str(url).strip()]

    return {
        "questionId": q.get("id"),
        "questionText": (q.get("questionText") or "").strip(),
        "answers": answer_texts,
        "currentCorrectIndices": current_correct_indices if current_correct_indices is not None else (q.get("correctIndices") or []),
        "explanationText": (q.get("explanationText") or "").strip(),
        "hasImages": bool(image_urls or image_refs),
        "imageRefs": image_refs,
        "imageUrls": image_urls,
    }
