
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any


REWARD_TYPE = "sequential"
REWARD_NAME = "tcm_final_outcome_only"

ALLOWED_SCHEMAS = {
    frozenset(("病因", "子类", "综合结论")),
    frozenset(("虚实", "寒热", "阴阳", "表里", "综合结论")),
    frozenset(("六经辨证证型", "综合结论")),
    frozenset(("气血津液辨证证型", "综合结论")),
    frozenset(("三焦辨证证型", "综合结论")),
    frozenset(("卫气营血证型", "综合结论")),
    frozenset(("脏腑辨证证型", "综合结论")),
}
SUMMARY_TEXT_FIELD = "综合结论"
SEPARATOR = re.compile(r"[,，、;；/／]")
OUTER = "\"'“”‘’[]【】"


def _json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def _final_summary(text: str) -> dict[str, Any] | None:
    if text.count("</reasoning>") != 1:
        return None
    tail = text.split("</reasoning>", maxsplit=1)[1]
    objects = _json_objects(tail)
    return objects[-1] if objects else None


def _component(value: str) -> str:
    result = unicodedata.normalize("NFKC", value)
    result = result.replace("淤", "瘀")
    result = re.sub(r"\s+", "", result)
    return result.strip(OUTER)


def _normalize(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, str):
        return None
    normalized = _component(value)
    if not normalized:
        return None
    parts = [_component(part) for part in SEPARATOR.split(normalized)]
    parts = [part for part in parts if part]
    if not parts:
        return None
    return tuple(sorted(dict.fromkeys(parts)))


def _score(response: str, ground_truth: str) -> dict[str, float]:
    predicted = _final_summary(response)
    expected = _final_summary(ground_truth)
    if expected is None or frozenset(expected) not in ALLOWED_SCHEMAS:
        raise ValueError("ground truth has an unsupported final-summary schema")

    expected_keys = set(expected)
    format_score = float(
        predicted is not None
        and set(predicted) == expected_keys
        and all(isinstance(predicted[key], str) for key in expected_keys)
    )
    categorical_fields = sorted(expected_keys - {SUMMARY_TEXT_FIELD})
    correct = 0
    if format_score:
        for field in categorical_fields:
            target_value = _normalize(expected[field])
            predicted_value = _normalize(predicted[field])
            correct += int(
                target_value is not None
                and predicted_value is not None
                and target_value == predicted_value
            )
    accuracy = correct / len(categorical_fields)
    overall = 0.10 * format_score + 0.90 * accuracy
    return {
        "overall": overall,
        "format": format_score,
        "accuracy": accuracy,
    }


def compute_score(data: dict[str, Any], **_: Any) -> dict[str, float]:
    responses = data.get("responses", data.get("response", ""))
    response = responses[0] if isinstance(responses, list) and responses else responses
    ground_truth = data.get("ground_truth") or data.get("output") or ""
    return _score(str(response), str(ground_truth))

