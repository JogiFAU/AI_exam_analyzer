"""Dataset cleanup helpers based on a JSON whitelist spec."""

from copy import deepcopy
from typing import Any, Dict

_DROP = object()


def _filter_by_rule(value: Any, rule: Any) -> Any:
    if rule is True:
        return deepcopy(value)

    if isinstance(rule, list):
        if not isinstance(value, dict):
            return _DROP
        return {k: deepcopy(value[k]) for k in rule if k in value}

    if not isinstance(rule, dict):
        return _DROP

    if isinstance(value, dict):
        wildcard_rule = rule.get("*")
        filtered: Dict[str, Any] = {}
        for key, item in value.items():
            chosen_rule = rule.get(key, wildcard_rule)
            if chosen_rule is None:
                continue
            out = _filter_by_rule(item, chosen_rule)
            if out is not _DROP:
                filtered[key] = out
        return filtered

    if isinstance(value, list):
        item_rule = rule.get("*")
        if item_rule is None:
            return []
        filtered_list = []
        for item in value:
            out = _filter_by_rule(item, item_rule)
            if out is not _DROP:
                filtered_list.append(out)
        return filtered_list

    return _DROP


def cleanup_dataset(data: Any, cleanup_spec: Dict[str, Any]) -> Any:
    """Return a cleaned copy of ``data`` that only keeps keys listed in ``cleanup_spec``."""
    cleaned = _filter_by_rule(data, cleanup_spec)
    if cleaned is _DROP:
        raise ValueError("Cleanup spec does not match dataset root type.")
    return cleaned
