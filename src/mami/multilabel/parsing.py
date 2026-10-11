"""Strict JSON parser: malformed outputs are invalid, not negative predictions."""
import json
from .prompts import LABELS


def parse_response(response):
    try:
        value = json.loads(response.strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != set(LABELS):
        return None
    if any(type(value[label]) is not int or value[label] not in (0, 1) for label in LABELS):
        return None
    return {label: value[label] for label in LABELS}
