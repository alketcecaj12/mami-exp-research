"""Strict JSON parser: malformed outputs are invalid, not negative predictions."""
import json
import re
from .prompts import LABELS


def parse_response(response):
    if not isinstance(response, str):
        return None
    raw = response.strip()
    if raw.startswith("```"):
        match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", raw, flags=re.IGNORECASE)
        if match is None:
            return None
        raw = match.group(1).strip()
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != set(LABELS):
        return None
    if any(type(value[label]) is not int or value[label] not in (0, 1) for label in LABELS):
        return None
    return {label: value[label] for label in LABELS}
