"""Core processing loop for question annotation."""

import time
from typing import Any, Dict, List, Optional

from ai_exam_analyzer.cleanup import cleanup_dataset
from ai_exam_analyzer.config import PIPELINE_VERSION
from ai_exam_analyzer.io_utils import save_json
from ai_exam_analyzer.knowledge_base import KnowledgeBase, build_query_text
from ai_exam_analyzer.passes import run_pass_a, run_pass_b, should_run_pass_b
from ai_exam_analyzer.payload import build_question_payload


def normalize_indices(indices: List[int], n_answers: int) -> List[int]:
    return sorted({i for i in indices if isinstance(i, int) and 0 <= i < n_answers})


def _build_output_obj(
    *,
    container: Optional[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    cleanup_spec: Optional[Dict[str, Any]],
) -> Any:
    out_obj: Any = container if container is not None else questions
    if cleanup_spec is not None:
        out_obj = cleanup_dataset(out_obj, cleanup_spec)
    return out_obj




def _compose_confidence(*, answer_conf: float, topic_conf: float, retrieval_quality: float, verifier_agreed: Optional[bool]) -> float:
    agreement = 1.0 if verifier_agreed is True else (0.5 if verifier_agreed is None else 0.25)
    score = (0.4 * answer_conf) + (0.25 * topic_conf) + (0.2 * retrieval_quality) + (0.15 * agreement)
    return max(0.0, min(1.0, round(score, 4)))


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


def process_questions(
    *,
    args: Any,
    questions: List[Dict[str, Any]],
    container: Optional[Dict[str, Any]],
    key_map: Dict[str, Dict[str, Any]],
    topic_catalog_text: str,
    schema_a: Dict[str, Any],
    schema_b: Dict[str, Any],
    cleanup_spec: Optional[Dict[str, Any]] = None,
    knowledge_base: Optional[KnowledgeBase] = None,
) -> None:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: install `openai` package (e.g. `pip install openai`).") from exc

    client = OpenAI()

    done = 0
    skipped = 0
    processed = 0

    for i, q in enumerate(questions, start=1):
        if args.limit and processed >= args.limit:
            break

        if args.resume and isinstance(q.get("aiAudit"), dict):
            if q["aiAudit"].get("pipelineVersion") == PIPELINE_VERSION and q["aiAudit"].get("status") == "completed":
                skipped += 1
                continue

        payload = build_question_payload(q)

        evidence_chunks: List[Dict[str, Any]] = []
        retrieval_quality = 0.0
        if knowledge_base is not None:
            evidence_chunks, retrieval_quality = knowledge_base.retrieve(
                build_query_text(payload),
                top_k=max(1, int(args.knowledge_top_k)),
                min_score=float(args.knowledge_min_score),
                max_chars=max(500, int(args.knowledge_max_chars)),
            )
            payload["retrievedEvidence"] = evidence_chunks

        answers = q.get("answers") or []
        n_answers = len(answers)
        current = normalize_indices(q.get("correctIndices") or [], n_answers)

        audit: Dict[str, Any] = {
            "pipelineVersion": PIPELINE_VERSION,
            "status": "error",
            "models": {"passA": args.passA_model, "passB": None},
            "knowledge": {
                "enabled": bool(knowledge_base is not None),
                "retrievalQuality": retrieval_quality,
                "evidenceCount": len(evidence_chunks),
            },
        }

        try:
            pass_a = run_pass_a(
                client,
                topic_catalog_text=topic_catalog_text,
                payload=payload,
                schema=schema_a,
                model=args.passA_model,
                temperature=args.passA_temperature,
            )

            proposed = normalize_indices(pass_a["answer_review"]["proposedCorrectIndices"], n_answers)

            final_topic_key = pass_a["topic_final"]["topicKey"]
            final_topic_conf = float(pass_a["topic_final"]["confidence"])
            final_topic_reason = pass_a["topic_final"]["reasonShort"]
            final_topic_source = "passA"

            maintenance = pass_a["maintenance"]
            extra_flags = pass_a["answer_review"].get("maintenanceSuspicion", []) or []
            merged_reasons = list(dict.fromkeys((maintenance.get("reasons") or []) + extra_flags))

            recommend_a = bool(pass_a["answer_review"]["recommendChange"])
            conf_a = float(pass_a["answer_review"]["confidence"])

            ai_disagrees_with_dataset = len(proposed) > 0 and proposed != current
            final_answer_confidence = conf_a
            final_answer_confidence_source = "passA"

            will_change = False
            change_source = "none"
            final_correct_indices = current
            verification: Dict[str, Any] = {"ran": False}
            verifier_agreed: Optional[bool] = None

            ran_b = should_run_pass_b(pass_a, args.trigger_answer_conf, args.trigger_topic_conf)
            pass_b: Optional[Dict[str, Any]] = None

            if ran_b:
                try:
                    pass_b = run_pass_b(
                        client,
                        topic_catalog_text=topic_catalog_text,
                        payload=payload,
                        pass_a=pass_a,
                        schema=schema_b,
                        model=args.passB_model,
                        reasoning_effort=args.passB_reasoning_effort,
                    )
                    audit["models"]["passB"] = args.passB_model

                    m_b = pass_b["maintenance"]
                    merged_reasons = list(dict.fromkeys(merged_reasons + (m_b.get("reasons") or [])))
                    maintenance = {
                        "needsMaintenance": bool(maintenance.get("needsMaintenance")) or bool(m_b.get("needsMaintenance")),
                        "severity": int(max(int(maintenance.get("severity", 1)), int(m_b.get("severity", 1)))),
                        "reasons": merged_reasons,
                    }

                    final_topic_key = pass_b["topic_final"]["topicKey"]
                    final_topic_conf = float(pass_b["topic_final"]["confidence"])
                    final_topic_reason = pass_b["topic_final"]["reasonShort"]
                    final_topic_source = "passB"

                    v = pass_b["verify_answer"]
                    cannot = bool(v.get("cannotJudge"))
                    agree = bool(v.get("agreeWithChange"))
                    conf_b = float(v.get("confidence"))
                    verified = normalize_indices(v.get("verifiedCorrectIndices", []), n_answers)

                    if len(verified) > 0 and verified != current:
                        ai_disagrees_with_dataset = True

                    final_answer_confidence = conf_b
                    final_answer_confidence_source = "passB"

                    verifier_agreed = agree and (not cannot)

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
                        "evidenceChunkIds": v.get("evidenceChunkIds", []),
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

            final_combined_confidence = _compose_confidence(
                answer_conf=final_answer_confidence,
                topic_conf=final_topic_conf,
                retrieval_quality=retrieval_quality,
                verifier_agreed=verifier_agreed,
            )

            if (
                (final_answer_confidence < args.low_conf_maintenance_threshold)
                or (final_topic_conf < args.low_conf_maintenance_threshold)
                or (final_combined_confidence < args.low_conf_maintenance_threshold)
            ):
                maintenance["needsMaintenance"] = True
                maintenance["severity"] = max(int(maintenance.get("severity", 1)), 2)
                maintenance["reasons"] = list(dict.fromkeys((maintenance.get("reasons") or []) + [
                    "low_confidence_answer_or_topic_or_combined"
                ]))

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
                        "evidenceChunkIds": pass_a["answer_review"].get("evidenceChunkIds", []),
                    },
                    "finalCorrectIndices": final_correct_indices,
                    "finalAnswerConfidence": final_answer_confidence,
                    "finalAnswerConfidenceSource": final_answer_confidence_source,
                    "finalCombinedConfidence": final_combined_confidence,
                    "retrievalQuality": retrieval_quality,
                    "evidenceCount": len(evidence_chunks),
                    "evidence": evidence_chunks,
                    "aiDisagreesWithDataset": ai_disagrees_with_dataset,
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
            out_obj = _build_output_obj(container=container, questions=questions, cleanup_spec=cleanup_spec)
            save_json(args.output, out_obj)
            print(f"[{i}/{len(questions)}] checkpoint | processed={processed} done={done} skipped={skipped} lastStatus={audit.get('status')}")

        time.sleep(args.sleep)

    out_obj = _build_output_obj(container=container, questions=questions, cleanup_spec=cleanup_spec)
    save_json(args.output, out_obj)
    print(f"Finished. processed={processed} done={done} skipped={skipped}. Output: {args.output}")
