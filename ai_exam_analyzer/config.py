"""Configuration defaults for the analyzer CLI."""

CONFIG = {
    "INPUT_PATH": "export.json",
    "TOPICS_PATH": "topic-tree.json",
    "OUTPUT_PATH": "export.AIannotated.json",
    "RESUME": False,
    "LIMIT": 0,
    "CHECKPOINT_EVERY": 10,
    "SLEEP": 0.15,
    "PASSA_MODEL": "gpt-4o-mini",
    "PASSB_MODEL": "o4-mini",
    "PASSA_TEMPERATURE": 0.0,
    "PASSB_REASONING_EFFORT": "high",
    "TRIGGER_ANSWER_CONF": 0.80,
    "TRIGGER_TOPIC_CONF": 0.85,
    "APPLY_CHANGE_MIN_CONF_B": 0.80,
    "LOW_CONF_MAINTENANCE_THRESHOLD": 0.65,
    "CLEANUP_SPEC_PATH": "",
    "IMAGES_ZIP_PATH": "images.zip",
    "KNOWLEDGE_ZIP_PATH": "",
    "KNOWLEDGE_INDEX_PATH": "",
    "KNOWLEDGE_SUBJECT_HINT": "",
    "KNOWLEDGE_TOP_K": 6,
    "KNOWLEDGE_MAX_CHARS": 4000,
    "KNOWLEDGE_MIN_SCORE": 0.06,
    "KNOWLEDGE_CHUNK_CHARS": 1200,
    "WRITE_TOP_LEVEL": True,
    "DEBUG": False,
}

PIPELINE_VERSION = "2pass-merged-v3-rag"
