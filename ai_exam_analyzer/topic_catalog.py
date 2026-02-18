"""Topic catalog construction and prompt formatting."""

from typing import Any, Dict, List, Tuple


def build_topic_catalog(topic_tree: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Build deterministic topic catalog and a topicKey->row map."""
    catalog: List[Dict[str, Any]] = []
    key_map: Dict[str, Dict[str, Any]] = {}

    super_topics = topic_tree.get("superTopics", [])
    if not isinstance(super_topics, list) or not super_topics:
        raise ValueError("Topic tree must have non-empty 'superTopics' array.")

    for s_idx, st in enumerate(super_topics, start=1):
        super_name = (st.get("name") or "").strip()
        subs = st.get("subtopics") or []
        if not isinstance(subs, list) or not subs:
            raise ValueError(f"SuperTopic '{super_name}' must have non-empty 'subtopics' list.")
        for sub_idx, sub in enumerate(subs, start=1):
            aliases: List[str] = []
            if isinstance(sub, dict):
                sub_name = (sub.get("name") or "").strip()
                raw_aliases = sub.get("aliases") or []
                if isinstance(raw_aliases, list):
                    aliases = [str(x).strip() for x in raw_aliases if str(x).strip()]
            else:
                sub_name = (sub or "").strip()
            topic_key = f"{s_idx}:{sub_idx}"
            row = {
                "superTopicId": s_idx,
                "superTopicName": super_name,
                "subtopicId": sub_idx,
                "subtopicName": sub_name,
                "aliases": aliases,
                "topicKey": topic_key,
            }
            catalog.append(row)
            key_map[topic_key] = row

    return catalog, key_map


def format_topic_catalog_for_prompt(catalog: List[Dict[str, Any]]) -> str:
    lines = ["Erlaubte Topics (wähle genau EINEN topicKey):"]
    current_super = None
    for row in catalog:
        if row["superTopicId"] != current_super:
            current_super = row["superTopicId"]
            lines.append(f'\n{row["superTopicId"]}. {row["superTopicName"]}')
        lines.append(f'  - topicKey {row["topicKey"]}: {row["subtopicName"]}')
    return "\n".join(lines)
