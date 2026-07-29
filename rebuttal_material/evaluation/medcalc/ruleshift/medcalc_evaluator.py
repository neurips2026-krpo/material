from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
from collections import defaultdict
from typing import Any, Iterable


INTEGER_CALCULATOR_IDS = {
    4,
    15,
    16,
    17,
    18,
    20,
    21,
    25,
    27,
    28,
    29,
    32,
    33,
    36,
    43,
    45,
    48,
    51,
}
DECIMAL_CALCULATOR_IDS = {8}
EXPECTED_ACTIONS = {1, 2, 3, 4}


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}")
    return records


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def balanced_object(text: str, start: int) -> str | None:
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


def parse_output(text: str) -> dict[str, Any]:
    issues: list[str] = []
    wrapper_valid = (
        text.count("<reasoning>") == 1
        and text.count("</reasoning>") == 1
        and text.find("<reasoning>") < text.find("</reasoning>")
    )
    if not wrapper_valid:
        issues.append("reasoning_wrapper")

    actions: dict[int, dict[str, Any]] = {}
    markers = list(
        re.finditer(
            r"(?m)^action(\d+)\s*:\s*(?:\[[^\]\r\n]*\]\s*:\s*)?",
            text,
        )
    )
    seen: set[int] = set()
    for marker in markers:
        index = int(marker.group(1))
        if index in seen:
            issues.append(f"duplicate_action{index}")
            continue
        seen.add(index)
        candidate = balanced_object(text, marker.end())
        if candidate is None:
            issues.append(f"unparseable_action{index}")
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            issues.append(f"invalid_json_action{index}")
            continue
        if not isinstance(parsed, dict):
            issues.append(f"nonobject_action{index}")
            continue
        actions[index] = parsed
    for missing in sorted(EXPECTED_ACTIONS - set(actions)):
        issues.append(f"missing_action{missing}")
    for unexpected in sorted(set(actions) - EXPECTED_ACTIONS):
        issues.append(f"unexpected_action{unexpected}")

    summary = None
    summary_markers = list(
        re.finditer(r"Clinical rule execution summary\s*:\s*", text)
    )
    if len(summary_markers) != 1:
        issues.append("summary_count")
    elif summary_markers:
        candidate = balanced_object(text, summary_markers[0].end())
        if candidate is None:
            issues.append("unparseable_summary")
        else:
            try:
                summary = json.loads(candidate)
            except json.JSONDecodeError:
                issues.append("invalid_json_summary")
            if summary is not None and not isinstance(summary, dict):
                issues.append("nonobject_summary")
                summary = None

    analysis_count = len(re.findall(r"<analysis>[\s\S]*?</analysis>", text))
    if analysis_count < 4:
        issues.append("analysis_count")
    return {
        "actions": actions,
        "summary": summary,
        "issues": sorted(set(issues)),
        "malformed": bool(issues),
    }


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def normalize_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = normalize_text(value)
        if text == "true":
            return True
        if text == "false":
            return False
        return text
    if isinstance(value, list):
        return tuple(normalize_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted(
                (normalize_text(key), normalize_value(item))
                for key, item in value.items()
            )
        )
    return normalize_text(value)


def entity_pairs(value: Any) -> set[tuple[str, str]]:
    if not isinstance(value, dict):
        return set()
    return {
        (
            normalize_text(key),
            json.dumps(
                normalize_value(item),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        for key, item in value.items()
    }


def safe_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    match = re.fullmatch(
        r'["\']?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))'
        r"\s*(?:points?|score)?\s*[\"']?",
        str(value).strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    number = float(match.group(1))
    return number if math.isfinite(number) else None


def exact_numeric(predicted: Any, ground_truth: Any) -> bool:
    pred = safe_number(predicted)
    target = safe_number(ground_truth)
    return (
        pred is not None
        and target is not None
        and math.isclose(pred, target, rel_tol=0.0, abs_tol=1e-9)
    )


def official_correct(
    predicted: Any,
    ground_truth: Any,
    calculator_id: int,
    lower_limit: Any,
    upper_limit: Any,
) -> bool:
    """Safe equivalent of official v1.0.7 correctness for the 19-rule subset."""
    pred = safe_number(predicted)
    target = safe_number(ground_truth)
    if pred is None or target is None:
        return False
    if calculator_id in INTEGER_CALCULATOR_IDS:
        return round(pred) == target
    if calculator_id in DECIMAL_CALCULATOR_IDS:
        lower = safe_number(lower_limit)
        upper = safe_number(upper_limit)
        return lower is not None and upper is not None and lower <= pred <= upper
    raise ValueError(f"Unsupported calculator ID in frozen subset: {calculator_id}")


def division(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def wilson_interval(correct: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return [
        (centre - margin) / denominator,
        (centre + margin) / denominator,
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        return {"n": 0}
    boolean_metrics = [
        "malformed",
        "entity_exact",
        "calculator_id_exact",
        "calculator_rule_exact",
        "trace_text_exact",
        "intermediate_exact_numeric",
        "intermediate_official_correct",
        "action4_exact_numeric",
        "action4_official_correct",
        "final_exact_numeric",
        "final_official_correct",
        "three_answer_locations_consistent",
    ]
    summary: dict[str, Any] = {"n": count}
    for metric in boolean_metrics:
        total_true = sum(bool(row[metric]) for row in rows)
        if metric == "malformed":
            summary["malformed_rate"] = total_true / count
        else:
            summary[metric] = total_true / count
    for metric in ("entity_precision", "entity_recall", "entity_f1"):
        summary[f"macro_{metric}"] = sum(float(row[metric]) for row in rows) / count

    true_positive = sum(row["entity_tp"] for row in rows)
    predicted_total = sum(row["entity_predicted"] for row in rows)
    expected_total = sum(row["entity_expected"] for row in rows)
    micro_precision = division(true_positive, predicted_total)
    micro_recall = division(true_positive, expected_total)
    summary["micro_entity_precision"] = micro_precision
    summary["micro_entity_recall"] = micro_recall
    summary["micro_entity_f1"] = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    correct = sum(bool(row["final_official_correct"]) for row in rows)
    summary["final_official_accuracy_wilson95"] = wilson_interval(correct, count)
    return summary


def response_from_prediction(record: dict[str, Any]) -> str:
    for key in ("response", "generated_text", "model_output", "output"):
        if key in record:
            return str(record[key])
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--condition", default="unspecified")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    references = read_jsonl(args.reference.resolve())
    predictions = read_jsonl(args.predictions.resolve())
    prediction_map: dict[str, dict[str, Any]] = {}
    duplicate_predictions: list[str] = []
    for prediction in predictions:
        sample_id = str(prediction.get("sample_id", ""))
        if sample_id in prediction_map:
            duplicate_predictions.append(sample_id)
        prediction_map[sample_id] = prediction

    details: list[dict[str, Any]] = []
    for reference in references:
        sample_id = reference["sample_id"]
        prediction = prediction_map.get(sample_id, {})
        response = response_from_prediction(prediction)
        parsed = parse_output(response)
        target = parse_output(reference["output"])
        expected_actions = target["actions"]
        predicted_actions = parsed["actions"]
        evaluation_target = reference["evaluation_target"]
        calculator_id = int(evaluation_target["calculator_id"])
        expected_answer = evaluation_target["answer"]

        predicted_entities = entity_pairs(
            predicted_actions.get(1, {}).get("extracted_variables")
        )
        expected_entities = entity_pairs(
            expected_actions[1].get("extracted_variables")
        )
        entity_tp = len(predicted_entities & expected_entities)
        entity_precision = division(entity_tp, len(predicted_entities))
        entity_recall = division(entity_tp, len(expected_entities))
        entity_f1 = (
            2 * entity_precision * entity_recall / (entity_precision + entity_recall)
            if entity_precision + entity_recall
            else float(not predicted_entities and not expected_entities)
        )

        action2 = predicted_actions.get(2, {})
        expected_action2 = expected_actions[2]
        calculator_id_exact = exact_numeric(
            action2.get("calculator_id"), calculator_id
        )
        calculator_name_exact = (
            normalize_text(action2.get("calculator_name", ""))
            == normalize_text(expected_action2.get("calculator_name", ""))
        )

        action3 = predicted_actions.get(3, {})
        intermediate_value = action3.get("computed_value")
        action4_value = predicted_actions.get(4, {}).get("answer")
        summary = parsed["summary"] if isinstance(parsed["summary"], dict) else {}
        final_value = summary.get("answer")

        lower_limit = evaluation_target["lower_limit"]
        upper_limit = evaluation_target["upper_limit"]
        intermediate_official = official_correct(
            intermediate_value,
            expected_answer,
            calculator_id,
            lower_limit,
            upper_limit,
        )
        action4_official = official_correct(
            action4_value,
            expected_answer,
            calculator_id,
            lower_limit,
            upper_limit,
        )
        final_official = official_correct(
            final_value,
            expected_answer,
            calculator_id,
            lower_limit,
            upper_limit,
        )
        three_values = [
            safe_number(intermediate_value),
            safe_number(action4_value),
            safe_number(final_value),
        ]
        three_consistent = (
            all(value is not None for value in three_values)
            and max(three_values) - min(three_values) <= 1e-9
        )

        detail = {
            "condition": prediction.get("condition", args.condition),
            "sample_id": sample_id,
            "prediction_present": bool(prediction),
            "calculator_id": calculator_id,
            "calculator_name": reference["metadata"]["calculator_name"],
            "category": reference["metadata"]["category"],
            "evaluation_split": reference["metadata"]["evaluation_split"],
            "malformed": parsed["malformed"],
            "parse_issues": parsed["issues"],
            "entity_exact": predicted_entities == expected_entities,
            "entity_precision": entity_precision,
            "entity_recall": entity_recall,
            "entity_f1": entity_f1,
            "entity_tp": entity_tp,
            "entity_predicted": len(predicted_entities),
            "entity_expected": len(expected_entities),
            "calculator_id_exact": calculator_id_exact,
            "calculator_name_exact": calculator_name_exact,
            "calculator_rule_exact": calculator_id_exact
            and calculator_name_exact,
            "trace_text_exact": normalize_text(
                action3.get("calculation_trace", "")
            )
            == normalize_text(
                expected_actions[3].get("calculation_trace", "")
            ),
            "intermediate_value": intermediate_value,
            "intermediate_exact_numeric": exact_numeric(
                intermediate_value, expected_answer
            ),
            "intermediate_official_correct": intermediate_official,
            "action4_value": action4_value,
            "action4_exact_numeric": exact_numeric(action4_value, expected_answer),
            "action4_official_correct": action4_official,
            "final_value": final_value,
            "ground_truth_answer": expected_answer,
            "lower_limit": lower_limit,
            "upper_limit": upper_limit,
            "final_exact_numeric": exact_numeric(final_value, expected_answer),
            "final_official_correct": final_official,
            "three_answer_locations_consistent": three_consistent,
            "prompt_tokens": prediction.get("prompt_tokens"),
            "completion_tokens": prediction.get("completion_tokens"),
            "finish_reason": prediction.get("finish_reason"),
        }
        details.append(detail)

    extra_predictions = sorted(set(prediction_map) - {row["sample_id"] for row in references})
    grouped: dict[str, Any] = {
        "overall": summarize(details),
        "evaluation_split": {},
        "category": {},
        "calculator": {},
    }
    for field, group_name in (
        ("evaluation_split", "evaluation_split"),
        ("category", "category"),
        ("calculator_id", "calculator"),
    ):
        values = sorted({row[field] for row in details}, key=str)
        grouped[group_name] = {
            str(value): summarize([row for row in details if row[field] == value])
            for value in values
        }

    token_values = [
        int(row["prompt_tokens"])
        for row in details
        if row["prompt_tokens"] is not None
    ]
    completion_values = [
        int(row["completion_tokens"])
        for row in details
        if row["completion_tokens"] is not None
    ]
    summary = {
        "status": (
            "PASS"
            if not duplicate_predictions
            and not extra_predictions
            and len(prediction_map) == len(references)
            else "INCOMPLETE_OR_INVALID_INPUT"
        ),
        "condition": args.condition,
        "reference_records": len(references),
        "prediction_records": len(predictions),
        "matched_predictions": sum(row["prediction_present"] for row in details),
        "duplicate_prediction_ids": duplicate_predictions,
        "extra_prediction_ids": extra_predictions,
        "metrics": grouped,
        "token_statistics": {
            "prompt_tokens_max": max(token_values) if token_values else None,
            "prompt_tokens_mean": (
                sum(token_values) / len(token_values) if token_values else None
            ),
            "completion_tokens_max": (
                max(completion_values) if completion_values else None
            ),
            "completion_tokens_mean": (
                sum(completion_values) / len(completion_values)
                if completion_values
                else None
            ),
        },
        "official_semantics": {
            "integer_calculator_ids": sorted(INTEGER_CALCULATOR_IDS),
            "decimal_tolerance_calculator_ids": sorted(DECIMAL_CALCULATOR_IDS),
            "unsafe_eval_used": False,
            "primary_final_source": "Clinical rule execution summary.answer",
        },
    }

    write_jsonl(output_dir / "detailed_results.jsonl", details)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_fields = list(details[0]) if details else []
    with (output_dir / "detailed_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in details:
            serialized = dict(row)
            serialized["parse_issues"] = "|".join(row["parse_issues"])
            writer.writerow(serialized)

    overall = summary["metrics"]["overall"]
    markdown = (
        "# MedCalc evaluation summary\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Condition: `{args.condition}`\n"
        f"- N: {overall['n']}\n"
        f"- Official final accuracy: {overall['final_official_correct']:.4f}\n"
        f"- Exact numeric accuracy: {overall['final_exact_numeric']:.4f}\n"
        f"- Entity exact: {overall['entity_exact']:.4f}\n"
        f"- Entity micro-F1: {overall['micro_entity_f1']:.4f}\n"
        f"- Calculator/rule exact: {overall['calculator_rule_exact']:.4f}\n"
        f"- Intermediate official accuracy: "
        f"{overall['intermediate_official_correct']:.4f}\n"
        f"- Trace text exact: {overall['trace_text_exact']:.4f}\n"
        f"- Malformed rate: {overall['malformed_rate']:.4f}\n"
    )
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

