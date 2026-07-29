#!/usr/bin/env python3
"""Four-action K-RPO reward for the MedCalc-Bench rule-execution task.

This module is intentionally self-contained. It does not import the independent
evaluation code. The public evaluator is stricter and lives in a separate file.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover - server environment has httpx
    httpx = None


REWARD_TYPE = "sequential"
REWARD_NAME = "medcalc_krpo_four_action"
EXPECTED_ACTIONS = (1, 2, 3, 4)


def _balanced_json_object(text: str, start: int) -> str | None:
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


def _parse_actions(text: str) -> tuple[dict[int, dict[str, Any]], bool]:
    actions: dict[int, dict[str, Any]] = {}
    valid = True
    pattern = re.compile(
        r"(?m)^action(\d+)\s*:\s*(?:\[[^\]\r\n]*\]\s*:\s*)?"
    )
    markers = list(pattern.finditer(text))
    seen: set[int] = set()
    for marker in markers:
        index = int(marker.group(1))
        if index in seen:
            valid = False
            continue
        seen.add(index)
        candidate = _balanced_json_object(text, marker.end())
        if candidate is None:
            valid = False
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            valid = False
            continue
        if not isinstance(value, dict):
            valid = False
            continue
        actions[index] = value
    if set(actions) != set(EXPECTED_ACTIONS):
        valid = False
    return actions, valid


def _parse_summary(text: str) -> tuple[dict[str, Any] | None, bool]:
    reasoning_end = text.find("</reasoning>")
    if reasoning_end < 0:
        return None, False
    tail = text[reasoning_end + len("</reasoning>") :]
    marker = re.search(r"Clinical rule execution summary\s*:\s*", tail)
    if marker is None:
        return None, False
    candidate = _balanced_json_object(tail, marker.end())
    if candidate is None:
        return None, False
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None, False
    return (parsed, isinstance(parsed, dict))


def _parse_response(text: str) -> dict[str, Any]:
    wrapper_valid = (
        text.count("<reasoning>") == 1
        and text.count("</reasoning>") == 1
        and text.find("<reasoning>") < text.find("</reasoning>")
    )
    actions, actions_valid = _parse_actions(text)
    summary, summary_valid = _parse_summary(text)
    analysis_count = len(re.findall(r"<analysis>[\s\S]*?</analysis>", text))
    analysis_valid = analysis_count >= 4
    format_score = (
        0.25 * float(wrapper_valid)
        + 0.50 * float(actions_valid)
        + 0.15 * float(summary_valid)
        + 0.10 * float(analysis_valid)
    )
    return {
        "wrapper_valid": wrapper_valid,
        "actions": actions,
        "actions_valid": actions_valid,
        "summary": summary,
        "summary_valid": summary_valid,
        "format_score": format_score,
    }


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _normalize_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        lowered = _normalize_text(value)
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return lowered
    if isinstance(value, list):
        return tuple(_normalize_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted(
                (_normalize_text(key), _normalize_value(item))
                for key, item in value.items()
            )
        )
    return _normalize_text(value)


def _entity_pairs(value: Any) -> set[tuple[str, str]]:
    if not isinstance(value, dict):
        return set()
    pairs = set()
    for key, item in value.items():
        canonical = json.dumps(
            _normalize_value(item),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        pairs.add((_normalize_text(key), canonical))
    return pairs


def _set_f1(predicted: set[Any], expected: set[Any]) -> float:
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    true_positive = len(predicted & expected)
    precision = true_positive / len(predicted)
    recall = true_positive / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _safe_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    match = re.fullmatch(
        r'["\']?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))'
        r"\s*(?:points?|score)?\s*[\"']?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    number = float(match.group(1))
    return number if math.isfinite(number) else None


def _numeric_exact(predicted: Any, expected: Any) -> float:
    pred_number = _safe_number(predicted)
    expected_number = _safe_number(expected)
    if pred_number is None or expected_number is None:
        return float(_normalize_text(predicted) == _normalize_text(expected))
    return float(math.isclose(pred_number, expected_number, rel_tol=0.0, abs_tol=1e-9))


class EmbeddingClient:
    """OpenAI-compatible embedding client with an in-process vector cache."""

    def __init__(
        self,
        api_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_url = (
            api_url
            or os.getenv(
                "EMBEDDING_API_URL",
                "https://REPLACE_WITH_OPENAI_COMPATIBLE_ENDPOINT/v1/embeddings",
            )
        ).rstrip("/")
        self.model = model or os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-8B")
        self.timeout = timeout
        self._cache: dict[str, list[float]] = {}
        self._client = httpx.Client(timeout=timeout) if httpx is not None else None
        self._warned = False

    def embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        if self._client is None:
            raise RuntimeError("httpx is unavailable")
        response = self._client.post(
            self.api_url,
            json={"input": text, "model": self.model},
        )
        response.raise_for_status()
        vector = response.json()["data"][0]["embedding"]
        self._cache[text] = vector
        if len(self._cache) > 5000:
            self._cache.clear()
            self._cache[text] = vector
        return vector

    def similarity(self, left: str, right: str) -> float:
        left = left.strip()
        right = right.strip()
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        try:
            first = self.embed(left)
            second = self.embed(right)
            dot = sum(a * b for a, b in zip(first, second))
            norm_a = math.sqrt(sum(value * value for value in first))
            norm_b = math.sqrt(sum(value * value for value in second))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            raw = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
            if raw >= 0.92:
                return 1.0
            if raw < 0.72:
                return max(0.0, raw * 0.1)
            return raw**2
        except Exception as exc:  # match submitted reward's fail-closed behavior
            if not self._warned:
                print(f"MedCalc embedding reward unavailable: {exc}")
                self._warned = True
            return 0.0


class MedCalcKRPOReward:
    def __init__(
        self,
        api_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.embedding = EmbeddingClient(api_url=api_url, model=model)

    def score(self, response: str, ground_truth: str) -> dict[str, float]:
        pred = _parse_response(response)
        target = _parse_response(ground_truth)
        zeros = {
            "overall": 0.0,
            "format": pred["format_score"],
            "action1_entities": 0.0,
            "action2_rule": 0.0,
            "action3_computation": 0.0,
            "action3_trace": 0.0,
            "action4_answer": 0.0,
            "summary": 0.0,
        }
        if not pred["wrapper_valid"] or not target["actions_valid"]:
            return zeros

        pred_actions = pred["actions"]
        target_actions = target["actions"]

        pred_entities = _entity_pairs(
            pred_actions.get(1, {}).get("extracted_variables")
        )
        target_entities = _entity_pairs(
            target_actions[1].get("extracted_variables")
        )
        entity_f1 = _set_f1(pred_entities, target_entities)
        entity_exact = float(pred_entities == target_entities)
        action1_score = 0.5 * entity_exact + 0.5 * entity_f1

        pred_action2 = pred_actions.get(2, {})
        target_action2 = target_actions[2]
        calculator_id_exact = _numeric_exact(
            pred_action2.get("calculator_id"),
            target_action2.get("calculator_id"),
        )
        calculator_name_exact = float(
            _normalize_text(pred_action2.get("calculator_name", ""))
            == _normalize_text(target_action2.get("calculator_name", ""))
        )
        action2_schema = float(
            set(pred_action2)
            == {"calculator_id", "calculator_name", "applicability"}
        )
        action2_score = (
            0.45 * calculator_id_exact
            + 0.45 * calculator_name_exact
            + 0.10 * action2_schema
        )

        pred_action3 = pred_actions.get(3, {})
        target_action3 = target_actions[3]
        computed_exact = _numeric_exact(
            pred_action3.get("computed_value"),
            target_action3.get("computed_value"),
        )
        trace_similarity = self.embedding.similarity(
            str(pred_action3.get("calculation_trace", "")),
            str(target_action3.get("calculation_trace", "")),
        )
        action3_schema = float(
            set(pred_action3) == {"calculation_trace", "computed_value"}
        )
        action3_score = (
            0.45 * computed_exact
            + 0.45 * trace_similarity
            + 0.10 * action3_schema
        )

        action4_score = _numeric_exact(
            pred_actions.get(4, {}).get("answer"),
            target_actions[4].get("answer"),
        )

        pred_summary = pred["summary"] if isinstance(pred["summary"], dict) else {}
        target_summary = target["summary"]
        summary_id = _numeric_exact(
            pred_summary.get("calculator_id"),
            target_summary.get("calculator_id"),
        )
        summary_answer = _numeric_exact(
            pred_summary.get("answer"),
            target_summary.get("answer"),
        )
        summary_schema = float(
            set(pred_summary) == {"calculator_id", "answer"}
        )
        summary_score = (
            0.45 * summary_id + 0.45 * summary_answer + 0.10 * summary_schema
        )

        action_mean = (
            action1_score + action2_score + action3_score + action4_score
        ) / 4
        overall = (
            0.10 * pred["format_score"]
            + 0.50 * action_mean
            + 0.40 * summary_score
        )
        return {
            "overall": max(0.0, min(1.0, overall)),
            "format": pred["format_score"],
            "action1_entities": action1_score,
            "action2_rule": action2_score,
            "action3_computation": computed_exact,
            "action3_trace": trace_similarity,
            "action4_answer": action4_score,
            "summary": summary_score,
        }


medcalc_reward = MedCalcKRPOReward()


def compute_score(data: dict[str, Any], **_: Any) -> dict[str, float]:
    response = data.get("response", data.get("responses", ""))
    if isinstance(response, list):
        response = response[0] if response else ""
    ground_truth = data.get("ground_truth") or data.get("output") or ""
    return medcalc_reward.score(str(response), str(ground_truth))

