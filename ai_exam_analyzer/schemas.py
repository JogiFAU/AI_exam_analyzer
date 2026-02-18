"""Structured output schemas for analysis passes."""

from typing import Any, Dict, List


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
                    "reasonDetailed": {"type": "string"},
                },
                "required": ["topicKey", "confidence", "reasonShort", "reasonDetailed"],
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
                    "reasonDetailed": {"type": "string"},
                    "maintenanceSuspicion": {"type": "array", "items": {"type": "string"}},
                    "evidenceChunkIds": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "isPlausible",
                    "confidence",
                    "recommendChange",
                    "proposedCorrectIndices",
                    "reasonShort",
                    "reasonDetailed",
                    "maintenanceSuspicion",
                    "evidenceChunkIds",
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
                    "reasonDetailed": {"type": "string"},
                },
                "required": ["topicKey", "confidence", "reasonShort", "reasonDetailed"],
                "additionalProperties": False,
            },
            "question_abstraction": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
        },
        "required": ["topic_initial", "answer_review", "maintenance", "topic_final", "question_abstraction"],
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
                    "reasonDetailed": {"type": "string"},
                    "cannotJudge": {"type": "boolean"},
                    "evidenceChunkIds": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["agreeWithChange", "verifiedCorrectIndices", "confidence", "reasonShort", "reasonDetailed", "cannotJudge", "evidenceChunkIds"],
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
                    "reasonDetailed": {"type": "string"},
                },
                "required": ["topicKey", "confidence", "reasonShort", "reasonDetailed"],
                "additionalProperties": False,
            },
        },
        "required": ["verify_answer", "maintenance", "topic_final"],
        "additionalProperties": False,
    }


def schema_review_pass(topic_keys: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "finalCorrectIndices": {"type": "array", "items": {"type": "integer", "minimum": 0}},
            "finalTopicKey": {"type": "string", "enum": topic_keys},
            "reviewComment": {"type": "string"},
            "recommendManualReview": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["finalCorrectIndices", "finalTopicKey", "reviewComment", "recommendManualReview", "confidence"],
        "additionalProperties": False,
    }



def schema_reconstruction_pass() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "isLikelyLegacyQuestion": {"type": "boolean"},
            "legacySignals": {"type": "array", "items": {"type": "string"}},
            "qualityClass": {"type": "string", "enum": ["high", "medium", "low"]},
            "reconstructedQuestion": {
                "type": "object",
                "properties": {
                    "questionText": {"type": "string"},
                    "answers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "answerIndex": {"type": "integer", "minimum": 1},
                                "text": {"type": "string"},
                            },
                            "required": ["answerIndex", "text"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["questionText", "answers"],
                "additionalProperties": False,
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string"},
            "reconstructionStrategy": {
                "type": "string",
                "enum": ["cluster_completion", "knowledge_completion", "no_completion_manual_review"],
            },
            "recommendManualReview": {"type": "boolean"},
        },
        "required": [
            "isLikelyLegacyQuestion",
            "legacySignals",
            "qualityClass",
            "reconstructedQuestion",
            "confidence",
            "reasoning",
            "reconstructionStrategy",
            "recommendManualReview",
        ],
        "additionalProperties": False,
    }


def schema_explainer_pass() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "correctnessExplanation": {"type": "string"},
            "wrongOptionExplanations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "answerIndex": {"type": "integer", "minimum": 1},
                        "whyWrong": {"type": "string"},
                    },
                    "required": ["answerIndex", "whyWrong"],
                    "additionalProperties": False,
                },
            },
            "contextualization": {"type": "string"},
        },
        "required": ["summary", "correctnessExplanation", "wrongOptionExplanations", "contextualization"],
        "additionalProperties": False,
    }


def schema_abstraction_cluster_refinement() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "removeQuestionIds": {"type": "array", "items": {"type": "string"}},
            "mergeIntoClusterId": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": {"type": "string"},
        },
        "required": ["removeQuestionIds", "mergeIntoClusterId", "confidence", "reason"],
        "additionalProperties": False,
    }
