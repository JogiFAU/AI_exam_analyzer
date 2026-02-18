"""Core processing loop for question annotation."""

import time
from typing import Any, Callable, Dict, List, Optional

from ai_exam_analyzer.cleanup import cleanup_dataset
from ai_exam_analyzer.config import PIPELINE_VERSION
from ai_exam_analyzer.io_utils import save_json
from ai_exam_analyzer.image_store import QuestionImageStore
from ai_exam_analyzer.knowledge_base import KnowledgeBase, build_query_text
from ai_exam_analyzer.passes import run_pass_a, run_pass_b, run_review_pass, should_run_pass_b
from ai_exam_analyzer.payload import build_question_payload
from ai_exam_analyzer.workflow_context import build_dataset_context, cluster_abstractions
from ai_exam_analyzer.decision_policy import compose_confidence, should_apply_pass_b_change, should_run_review_pass
from ai_exam_analyzer.preprocessing import compute_preprocessing_assessment


def _answer_external_indices(q: Dict[str, Any]) -> List[int]:
    answers = q.get("answers") or []
    out: List[int] = []
    for i, answer in enumerate(answers):
        idx = None
        for key in ("answerIndex", "position", "index"):
            value = answer.get(key)
            if isinstance(value, int) and value > 0:
                idx = value
                break
        out.append(idx if idx is not None else (i + 1))
    return out


def normalize_indices(indices: List[int], n_answers: int, *, valid_indices: Optional[List[int]] = None) -> List[int]:
    valid_set = set(valid_indices or [])
    out: List[int] = []
    for i in (indices or []):
        if not isinstance(i, int):
            continue
        if valid_set:
            if i in valid_set:
                out.append(i)
        elif 0 <= i < n_answers:
            out.append(i)
    return sorted(set(out))


def _coerce_dataset_correct_indices(raw_indices: List[int], external_indices: List[int]) -> List[int]:
    """Support both legacy 0-based indices and new external 1-based answerIndex values."""
    n_answers = len(external_indices)
    if not raw_indices:
        return []
    cleaned = [i for i in raw_indices if isinstance(i, int)]
    if not cleaned:
        return []

    ext_set = set(external_indices)
    if all(i in ext_set for i in cleaned):
        return sorted(set(cleaned))

    if all(0 <= i < n_answers for i in cleaned):
        mapped = [external_indices[i] for i in cleaned]
        return sorted(set(mapped))

    return sorted({i for i in cleaned if i in ext_set})


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




def _compact_evidence(evidence_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for row in evidence_chunks:
        compact.append({
            "chunkId": row.get("chunkId"),
            "source": row.get("source"),
            "page": row.get("page"),
            "score": row.get("score"),
        })
    return compact


def apply_correct_indices(q: Dict[str, Any], new_indices: List[int]) -> None:
    """Update correctIndices + answers[].isCorrect + correctAnswers using external answer indices."""
    answers = q.get("answers") or []
    external_indices = _answer_external_indices(q)
    new_set = set(new_indices)

    for i, a in enumerate(answers):
        ext_idx = external_indices[i] if i < len(external_indices) else (i + 1)
        a["isCorrect"] = (ext_idx in new_set)

    q["correctIndices"] = sorted(new_set)
    correct_answers: List[Dict[str, Any]] = []
    for i, a in enumerate(answers):
        ext_idx = external_indices[i] if i < len(external_indices) else (i + 1)
        if ext_idx in new_set:
            correct_answers.append({"index": ext_idx, "text": a.get("text"), "html": a.get("html")})
    q["correctAnswers"] = correct_answers


def process_questions(
    *,
    args: Any,
    questions: List[Dict[str, Any]],
    container: Optional[Dict[str, Any]],
    key_map: Dict[str, Dict[str, Any]],
    topic_catalog_text: str,
    schema_a: Dict[str, Any],
    schema_b: Dict[str, Any],
    schema_review: Dict[str, Any],
    cleanup_spec: Optional[Dict[str, Any]] = None,
    knowledge_base: Optional[KnowledgeBase] = None,
    image_store: Optional[QuestionImageStore] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: install `openai` package (e.g. `pip install openai`).") from exc

    client = OpenAI()

    done = 0
    skipped = 0
    processed = 0
    total_questions = len(questions)

    def emit_progress(**payload: Any) -> None:
        if progress_callback is None:
            return
        progress_callback(payload)

    emit_progress(
        event="started",
        processed=processed,
        done=done,
        skipped=skipped,
        total=total_questions,
        message="Analyse gestartet.",
    )

    dataset_context = build_dataset_context(
        questions,
        image_store=image_store,
        knowledge_base=knowledge_base,
        text_similarity_threshold=float(args.text_cluster_similarity),
    )

    for i, q in enumerate(questions, start=1):
        if args.limit and processed >= args.limit:
            break

        if args.resume and isinstance(q.get("aiAudit"), dict):
            if q["aiAudit"].get("pipelineVersion") == PIPELINE_VERSION and q["aiAudit"].get("status") == "completed":
                skipped += 1
                emit_progress(
                    event="question_skipped",
                    index=i,
                    total=total_questions,
                    processed=processed,
                    done=done,
                    skipped=skipped,
                    message=f"Frage {i}/{total_questions} übersprungen (bereits abgeschlossen).",
                )
                continue

        external_indices = _answer_external_indices(q)
        current = _coerce_dataset_correct_indices(q.get("correctIndices") or [], external_indices)
        payload = build_question_payload(q, current_correct_indices=current)

        question_images: List[Dict[str, Any]] = []
        image_context: Dict[str, Any] = {"imageZipConfigured": bool(image_store is not None)}
        if image_store is not None:
            question_images, image_context = image_store.prepare_question_images(q)
        payload["imageContext"] = image_context
        qid = str(q.get("id") or "")
        payload["questionClusterContext"] = {
            "clusterId": dataset_context.text_clusters["questionToCluster"].get(qid),
            "clusterMembers": dataset_context.text_clusters["clusterMembers"].get(str(dataset_context.text_clusters["questionToCluster"].get(qid)), []),
        }
        question_image_clusters = ((dataset_context.image_clusters.get("questionImageClusters") or {}).get("questionToClusters") or {}).get(qid, [])
        all_image_clusters = (dataset_context.image_clusters.get("questionImageClusters") or {}).get("clusters", [])
        payload["imageClusterContext"] = {
            "clusterIds": question_image_clusters,
            "clusters": [c for c in all_image_clusters if c.get("clusterId") in set(question_image_clusters)],
        }
        payload["knowledgeImageContext"] = (dataset_context.image_clusters.get("knowledgeImageMatches") or {}).get(qid, [])

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
        preprocessing = compute_preprocessing_assessment(q)
        pre_maintenance_reasons = preprocessing.get("reasons", [])

        audit: Dict[str, Any] = {
            "pipelineVersion": PIPELINE_VERSION,
            "status": "error",
            "models": {"passA": args.passA_model, "passB": None, "review": None},
            "knowledge": {
                "enabled": bool(knowledge_base is not None),
                "retrievalQuality": retrieval_quality,
                "evidenceCount": len(evidence_chunks),
            },
            "images": image_context,
            "clusters": {
                "questionContentClusterId": payload["questionClusterContext"].get("clusterId"),
                "questionImageClusterIds": payload["imageClusterContext"].get("clusterIds", []),
            },
            "preprocessing": preprocessing,
        }

        try:
            if not bool((preprocessing.get("gates") or {}).get("runLlm", True)):
                maintenance = {
                    "needsMaintenance": True,
                    "severity": 3,
                    "reasons": list(dict.fromkeys(pre_maintenance_reasons + ["preprocessing_llm_skipped"])),
                }
                audit.update({
                    "status": "completed",
                    "topicInitial": {"superTopic": "", "subtopic": "", "confidence": 0.0, "reasonShort": "Skipped by preprocessing gate", "reasonDetailed": "runLlm=false"},
                    "topicFinal": {"superTopic": "", "subtopic": "", "confidence": 0.0, "reasonShort": "Skipped by preprocessing gate", "reasonDetailed": "runLlm=false", "source": "preprocessing"},
                    "answerPlausibility": {
                        "originalCorrectIndices": current,
                        "passA": {"isPlausible": False, "confidence": 0.0, "recommendChange": False, "proposedCorrectIndices": [], "reasonShort": "Skipped by preprocessing gate", "reasonDetailed": "runLlm=false", "evidenceChunkIds": []},
                        "finalCorrectIndices": current,
                        "finalAnswerConfidence": 0.0,
                        "finalAnswerConfidenceSource": "preprocessing",
                        "finalCombinedConfidence": 0.0,
                        "retrievalQuality": retrieval_quality,
                        "evidenceCount": len(evidence_chunks),
                        "evidence": _compact_evidence(evidence_chunks),
                        "aiDisagreesWithDataset": False,
                        "changedInDataset": False,
                        "changeSource": "none",
                        "verification": {"ran": False, "skippedByPreprocessing": True},
                    },
                    "maintenance": maintenance,
                    "questionAbstraction": {"summary": ""},
                })
                done += 1
                q["aiAudit"] = audit
                processed += 1
                emit_progress(event="question_finished", index=i, total=total_questions, processed=processed, done=done, skipped=skipped, status=audit.get("status"), message=f"Frage {i}/{total_questions} abgeschlossen (preprocessing skip).")
                if args.checkpoint_every and processed % args.checkpoint_every == 0:
                    out_obj = _build_output_obj(container=container, questions=questions, cleanup_spec=cleanup_spec)
                    save_json(args.output, out_obj)
                time.sleep(args.sleep)
                continue

            emit_progress(
                event="question_started",
                index=i,
                total=total_questions,
                processed=processed,
                done=done,
                skipped=skipped,
                message=f"Frage {i}/{total_questions}: Starte Pass A.",
            )
            pass_a = run_pass_a(
                client,
                topic_catalog_text=topic_catalog_text,
                payload=payload,
                schema=schema_a,
                model=args.passA_model,
                temperature=args.passA_temperature,
                question_images=question_images,
            )

            proposed = normalize_indices(
                pass_a["answer_review"].get("proposedCorrectIndices", []),
                n_answers,
                valid_indices=external_indices,
            )

            final_topic_key = pass_a["topic_final"]["topicKey"]
            final_topic_conf = float(pass_a["topic_final"]["confidence"])
            final_topic_reason = pass_a["topic_final"]["reasonShort"]
            final_topic_reason_detailed = pass_a["topic_final"]["reasonDetailed"]
            final_topic_source = "passA"

            maintenance = pass_a["maintenance"]
            extra_flags = pass_a["answer_review"].get("maintenanceSuspicion", []) or []
            merged_reasons = list(dict.fromkeys(pre_maintenance_reasons + (maintenance.get("reasons") or []) + extra_flags))
            if pre_maintenance_reasons:
                maintenance["needsMaintenance"] = True
                maintenance["severity"] = max(int(maintenance.get("severity", 1)), 2)

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
                    emit_progress(
                        event="pass_b_started",
                        index=i,
                        total=total_questions,
                        processed=processed,
                        done=done,
                        skipped=skipped,
                        message=f"Frage {i}/{total_questions}: Starte Verifikation (Pass B).",
                    )
                    pass_b = run_pass_b(
                        client,
                        topic_catalog_text=topic_catalog_text,
                        payload=payload,
                        pass_a=pass_a,
                        schema=schema_b,
                        model=args.passB_model,
                        reasoning_effort=args.passB_reasoning_effort,
                        question_images=question_images,
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
                    final_topic_reason_detailed = pass_b["topic_final"]["reasonDetailed"]
                    final_topic_source = "passB"

                    v = pass_b["verify_answer"]
                    cannot = bool(v.get("cannotJudge"))
                    agree = bool(v.get("agreeWithChange"))
                    conf_b = float(v.get("confidence"))
                    verified = normalize_indices(
                        v.get("verifiedCorrectIndices", []),
                        n_answers,
                        valid_indices=external_indices,
                    )

                    if len(verified) > 0 and verified != current:
                        ai_disagrees_with_dataset = True

                    final_answer_confidence = conf_b
                    final_answer_confidence_source = "passB"

                    verifier_agreed = agree and (not cannot)

                    if should_apply_pass_b_change(
                        current_indices=current,
                        verified_indices=verified,
                        cannot_judge=cannot,
                        agree_with_change=agree,
                        confidence_b=conf_b,
                        apply_min_conf_b=args.apply_change_min_conf_b,
                        retrieval_quality=retrieval_quality,
                        evidence_count=len(evidence_chunks),
                        allow_auto_change=bool((preprocessing.get("gates") or {}).get("allowAutoChange", True)),
                    ):
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
                        "reasonShort": v.get("reasonShort", ""),
                        "reasonDetailed": v.get("reasonDetailed", ""),
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

            final_combined_confidence = compose_confidence(
                answer_conf=final_answer_confidence,
                topic_conf=final_topic_conf,
                retrieval_quality=retrieval_quality,
                verifier_agreed=verifier_agreed,
                evidence_count=len(evidence_chunks),
                knowledge_enabled=bool(knowledge_base is not None),
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

            if bool((preprocessing.get("gates") or {}).get("forceManualReview", False)):
                maintenance["needsMaintenance"] = True
                maintenance["severity"] = max(int(maintenance.get("severity", 1)), 3)
                maintenance["reasons"] = list(dict.fromkeys((maintenance.get("reasons") or []) + ["preprocessing_force_manual_review"]))

            init_row = key_map[pass_a["topic_initial"]["topicKey"]]
            final_row = key_map[final_topic_key]
            compact_evidence = _compact_evidence(evidence_chunks)

            audit.update({
                "status": "completed",
                "topicInitial": {
                    "superTopic": init_row["superTopicName"],
                    "subtopic": init_row["subtopicName"],
                    "confidence": float(pass_a["topic_initial"]["confidence"]),
                    "reasonShort": pass_a["topic_initial"]["reasonShort"],
                    "reasonDetailed": pass_a["topic_initial"]["reasonDetailed"],
                },
                "topicFinal": {
                    "superTopic": final_row["superTopicName"],
                    "subtopic": final_row["subtopicName"],
                    "confidence": final_topic_conf,
                    "reasonShort": final_topic_reason,
                    "reasonDetailed": final_topic_reason_detailed,
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
                        "reasonDetailed": pass_a["answer_review"]["reasonDetailed"],
                        "evidenceChunkIds": pass_a["answer_review"].get("evidenceChunkIds", []),
                    },
                    "finalCorrectIndices": final_correct_indices,
                    "finalAnswerConfidence": final_answer_confidence,
                    "finalAnswerConfidenceSource": final_answer_confidence_source,
                    "finalCombinedConfidence": final_combined_confidence,
                    "retrievalQuality": retrieval_quality,
                    "evidenceCount": len(evidence_chunks),
                    "evidence": compact_evidence,
                    "aiDisagreesWithDataset": ai_disagrees_with_dataset,
                    "changedInDataset": bool(will_change),
                    "changeSource": change_source,
                    "verification": verification,
                },
                "maintenance": maintenance,
                "questionAbstraction": {
                    "summary": (pass_a.get("question_abstraction") or {}).get("summary", ""),
                },
            })

            force_manual_review = bool((preprocessing.get("gates") or {}).get("forceManualReview", False))
            if force_manual_review or should_run_review_pass(
                args=args,
                maintenance=audit.get("maintenance", {}),
                ai_disagrees_with_dataset=ai_disagrees_with_dataset,
                final_combined_confidence=final_combined_confidence,
                pass_a_topic_key=pass_a["topic_initial"]["topicKey"],
                final_topic_key=final_topic_key,
            ):
                try:
                    review = run_review_pass(
                        client,
                        payload=payload,
                        current_audit=audit,
                        schema=schema_review,
                        model=args.review_model,
                        question_images=question_images,
                    )
                    audit["models"]["review"] = args.review_model
                    review_indices = normalize_indices(
                        review.get("finalCorrectIndices", []),
                        n_answers,
                        valid_indices=external_indices,
                    )
                    if review_indices and review_indices != (q.get("correctIndices") or []):
                        apply_correct_indices(q, review_indices)
                        audit["answerPlausibility"]["finalCorrectIndices"] = review_indices
                    topic_key_review = review.get("finalTopicKey")
                    if topic_key_review in key_map:
                        topic_row_review = key_map[topic_key_review]
                        audit["topicFinal"]["superTopic"] = topic_row_review["superTopicName"]
                        audit["topicFinal"]["subtopic"] = topic_row_review["subtopicName"]
                        audit["topicFinal"]["source"] = "review"
                        audit["topicFinal"]["confidence"] = float(review.get("confidence", audit["topicFinal"].get("confidence", 0.0)))
                        audit["topicFinal"]["reasonShort"] = "Pass-C review override"
                        audit["topicFinal"]["reasonDetailed"] = review.get("reviewComment", "")
                    audit["reviewPass"] = review
                    if review.get("recommendManualReview"):
                        audit["maintenance"]["needsMaintenance"] = True
                        audit["maintenance"]["reasons"] = list(dict.fromkeys((audit["maintenance"].get("reasons") or []) + ["review_pass_manual_review"]))
                except Exception as review_exc:
                    audit["models"]["review"] = args.review_model
                    audit["reviewPass"] = {"error": str(review_exc)}

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
        emit_progress(
            event="question_finished",
            index=i,
            total=total_questions,
            processed=processed,
            done=done,
            skipped=skipped,
            status=audit.get("status"),
            message=f"Frage {i}/{total_questions} abgeschlossen (Status: {audit.get('status')}).",
        )
        if args.checkpoint_every and processed % args.checkpoint_every == 0:
            out_obj = _build_output_obj(container=container, questions=questions, cleanup_spec=cleanup_spec)
            save_json(args.output, out_obj)
            print(f"[{i}/{len(questions)}] checkpoint | processed={processed} done={done} skipped={skipped} lastStatus={audit.get('status')}")
            emit_progress(
                event="checkpoint_saved",
                index=i,
                total=total_questions,
                processed=processed,
                done=done,
                skipped=skipped,
                status=audit.get("status"),
                message=f"Checkpoint gespeichert ({processed} verarbeitet).",
            )

        time.sleep(args.sleep)

    abstraction_clusters = cluster_abstractions(
        questions,
        threshold=float(args.abstraction_cluster_similarity),
    )
    for q in questions:
        qid = str(q.get("id") or "")
        audit = q.get("aiAudit")
        if not isinstance(audit, dict):
            continue
        audit.setdefault("clusters", {})
        audit["clusters"]["abstractionClusterId"] = abstraction_clusters["questionToAbstractionCluster"].get(qid)

    out_obj = _build_output_obj(container=container, questions=questions, cleanup_spec=cleanup_spec)
    save_json(args.output, out_obj)
    print(f"Finished. processed={processed} done={done} skipped={skipped}. Output: {args.output}")
    emit_progress(
        event="finished",
        processed=processed,
        done=done,
        skipped=skipped,
        total=total_questions,
        output_path=args.output,
        message=f"Analyse abgeschlossen. Output: {args.output}",
    )
