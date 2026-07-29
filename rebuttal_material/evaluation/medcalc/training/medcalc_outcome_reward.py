#!/usr/bin/env python3
"""Final-answer-only GRPO reward for the matched MedCalc baseline.

No intermediate action content contributes to this reward.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any


REWARD_TYPE = "sequential"
REWARD_NAME = "medcalc_outcome_only"


def _balanced_object(text: str, start: int) -> str | None:
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _summary(text: str) -> dict[str, Any] | None:
    marker = re.search(r"Clinical rule execution summary\s*:\s*", text)
    if marker is None:
        return None
    candidate = _balanced_object(text, marker.end())
    if candidate is None:
        return None
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    match = re.fullmatch(
        r'["\']?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))'
        r"\s*(?:points?|score)?\s*[\"']?",
        str(value).strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = float(match.group(1))
    return value if math.isfinite(value) else None


def _exact(left: Any, right: Any) -> float:
    first = _number(left)
    second = _number(right)
    if first is None or second is None:
        return 0.0
    return float(math.isclose(first, second, rel_tol=0.0, abs_tol=1e-9))


def compute_score(data: dict[str, Any], **_: Any) -> dict[str, float]:
    response = data.get("response", data.get("responses", ""))
    if isinstance(response, list):
        response = response[0] if response else ""
    ground_truth = data.get("ground_truth") or data.get("output") or ""
    predicted = _summary(str(response))
    expected = _summary(str(ground_truth))
    format_score = float(
        predicted is not None and set(predicted) == {"calculator_id", "answer"}
    )
    accuracy = (
        _exact(predicted.get("answer"), expected.get("answer"))
        if predicted is not None and expected is not None
        else 0.0
    )
    overall = 0.10 * format_score + 0.90 * accuracy
    return {
        "overall": overall,
        "format": format_score,
        "accuracy": accuracy,
    }

