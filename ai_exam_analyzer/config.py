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
    "WRITE_TOP_LEVEL": True,
    "DEBUG": False,
}

PIPELINE_VERSION = "2pass-merged-v2"
