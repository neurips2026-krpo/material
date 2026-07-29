#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import random
import re
from collections import defaultdict
from typing import Any, Callable


CONDITIONS = (
    "base_ruleshift",
    "outcome_ruleshift",
    "krpo_ruleshift",
)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260727


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def load_module(path: pathlib.Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_key(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    match = re.fullmatch(
        r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"(?:\s*[a-zA-Z%/^0-9._-]+(?:\s+[a-zA-Z%/^0-9._-]+)*)?\s*",
        str(value),
    )
    if not match:
        return None
    result = float(match.group(1))
    return result if math.isfinite(result) else None


def canonical_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        if value:
            parsed = numeric_value(value[0])
            if parsed is not None:
                return parsed
        return tuple(canonical_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted(
                (normalize_key(key), canonical_value(item))
                for key, item in value.items()
            )
        )
    text = normalize_key(value)
    if text == "true":
        return True
    if text == "false":
        return False
    parsed = numeric_value(text)
    return parsed if parsed is not None else text


def canonical_entities(
    value: Any, aliases: dict[str, str]
) -> tuple[set[str], set[tuple[str, str]]]:
    if not isinstance(value, dict):
        return set(), set()
    keys = set()
    pairs = set()
    for raw_key, raw_value in value.items():
        normalized = normalize_key(raw_key)
        key = aliases.get(normalized, normalized)
        keys.add(key)
        encoded = json.dumps(
            canonical_value(raw_value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        pairs.add((key, encoded))
    return keys, pairs


def f1(predicted: set[Any], expected: set[Any]) -> float:
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = len(predicted & expected)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap(
    candidate: list[float], comparator: list[float], seed: int
) -> list[float]:
    if len(candidate) != len(comparator) or not candidate:
        raise ValueError("paired bootstrap inputs are invalid")
    differences = [
        left - right for left, right in zip(candidate, comparator)
    ]
    generator = random.Random(seed)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        samples.append(
            sum(
                differences[generator.randrange(len(differences))]
                for _ in differences
            )
            / len(differences)
        )
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def exact_mcnemar(
    candidate: list[bool], comparator: list[bool]
) -> dict[str, Any]:
    candidate_only = sum(
        left and not right for left, right in zip(candidate, comparator)
    )
    comparator_only = sum(
        right and not left for left, right in zip(candidate, comparator)
    )
    discordant = candidate_only + comparator_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(candidate_only, comparator_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "candidate_only": candidate_only,
        "comparator_only": comparator_only,
        "discordant": discordant,
        "p_value_raw": p_value,
    }


def holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        current = min(1.0, (len(values) - rank) * values[index])
        running = max(running, current)
        adjusted[index] = running
    return adjusted


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    keys = (
        "control_correct",
        "revision_correct",
        "stale_original",
        "switch_success",
        "revision_action3_correct",
        "revision_summary_valid",
        "revision_final_and_pair_f1_ge_80",
    )
    result: dict[str, Any] = {"n": n}
    for key in keys:
        result[key] = sum(bool(row[key]) for row in rows) / n
    result["control_pair_f1"] = sum(row["control_pair_f1"] for row in rows) / n
    result["revision_pair_f1"] = sum(row["revision_pair_f1"] for row in rows) / n
    result["revision_completion_tokens"] = (
        sum(row["revision_completion_tokens"] for row in rows) / n
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task10-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.task10_root.resolve()
    evaluator = load_module(
        root / "medcalc_evaluator.py", "task10_medcalc_evaluator"
    )
    cards = json.loads(
        (root / "medcalc_rule_cards.json").read_text(encoding="utf-8")
    )["cards"]
    aliases_by_calculator: dict[int, dict[str, str]] = {}
    for card in cards:
        calculator_id = int(card["calculator_id"])
        if calculator_id not in {4, 8, 18, 45}:
            continue
        aliases = {}
        for display_name, implementation_name in card["required_inputs"].items():
            canonical = normalize_key(implementation_name)
            aliases[normalize_key(display_name)] = canonical
            aliases[canonical] = canonical
        aliases_by_calculator[calculator_id] = aliases

    targets = read_jsonl(
        root / "targets/medcalc_ruleshift_targets_67.jsonl"
    )
    predictions = {
        condition: {
            row["sample_id"]: row
            for row in read_jsonl(
                root
                / f"server_runs/predictions/{condition}/predictions.jsonl"
            )
        }
        for condition in CONDITIONS
    }
    expected_ids = {
        sample_id
        for target in targets
        for sample_id in (
            target["control_sample_id"],
            target["revision_sample_id"],
        )
    }
    for condition in CONDITIONS:
        if set(predictions[condition]) != expected_ids:
            raise ValueError(f"{condition}: prediction IDs do not match targets")

    case_rows = []
    for condition in CONDITIONS:
        for target in targets:
            calculator_id = int(target["metadata"]["calculator_id"])
            aliases = aliases_by_calculator[calculator_id]
            expected_keys, expected_pairs = canonical_entities(
                target["expected_extracted_variables"], aliases
            )
            parsed = {}
            for variant, sample_id in (
                ("control", target["control_sample_id"]),
                ("revision", target["revision_sample_id"]),
            ):
                prediction = predictions[condition][sample_id]
                output = evaluator.parse_output(str(prediction["response"]))
                summary = (
                    output["summary"]
                    if isinstance(output["summary"], dict)
                    else {}
                )
                final_answer = summary.get("answer")
                action3_answer = output["actions"].get(3, {}).get(
                    "computed_value"
                )
                entities = output["actions"].get(1, {}).get(
                    "extracted_variables"
                )
                keys, pairs = canonical_entities(entities, aliases)
                parsed[variant] = {
                    "final_answer": final_answer,
                    "action3_answer": action3_answer,
                    "summary_valid": (
                        set(summary) == {"calculator_id", "answer"}
                        and evaluator.exact_numeric(
                            summary.get("calculator_id"), calculator_id
                        )
                    ),
                    "key_f1": f1(keys, expected_keys),
                    "pair_f1": f1(pairs, expected_pairs),
                    "completion_tokens": int(prediction["completion_tokens"]),
                    "issues": output["issues"],
                }
            original = target["original_answer"]
            revised = target["revised_answer"]
            case_rows.append(
                {
                    "condition": condition,
                    "base_sample_id": target["base_sample_id"],
                    "calculator_id": calculator_id,
                    "criterion_id": target["metadata"]["criterion_id"],
                    "control_correct": evaluator.exact_numeric(
                        parsed["control"]["final_answer"], original
                    ),
                    "revision_correct": evaluator.exact_numeric(
                        parsed["revision"]["final_answer"], revised
                    ),
                    "stale_original": evaluator.exact_numeric(
                        parsed["revision"]["final_answer"], original
                    ),
                    "switch_success": (
                        evaluator.exact_numeric(
                            parsed["control"]["final_answer"], original
                        )
                        and evaluator.exact_numeric(
                            parsed["revision"]["final_answer"], revised
                        )
                    ),
                    "revision_action3_correct": evaluator.exact_numeric(
                        parsed["revision"]["action3_answer"], revised
                    ),
                    "revision_summary_valid": parsed["revision"]["summary_valid"],
                    "control_key_f1": parsed["control"]["key_f1"],
                    "control_pair_f1": parsed["control"]["pair_f1"],
                    "revision_key_f1": parsed["revision"]["key_f1"],
                    "revision_pair_f1": parsed["revision"]["pair_f1"],
                    "revision_final_and_pair_f1_ge_80": (
                        evaluator.exact_numeric(
                            parsed["revision"]["final_answer"], revised
                        )
                        and parsed["revision"]["pair_f1"] >= 0.8
                    ),
                    "control_final_answer": parsed["control"]["final_answer"],
                    "revision_final_answer": parsed["revision"]["final_answer"],
                    "original_answer": original,
                    "revised_answer": revised,
                    "control_completion_tokens": parsed["control"][
                        "completion_tokens"
                    ],
                    "revision_completion_tokens": parsed["revision"][
                        "completion_tokens"
                    ],
                    "control_issues": parsed["control"]["issues"],
                    "revision_issues": parsed["revision"]["issues"],
                }
            )

    rows_by_condition = {
        condition: [
            row for row in case_rows if row["condition"] == condition
        ]
        for condition in CONDITIONS
    }
    summary = {
        condition: summarize(rows)
        for condition, rows in rows_by_condition.items()
    }
    by_calculator = {}
    for calculator_id in (4, 8, 18, 45):
        by_calculator[str(calculator_id)] = {
            condition: summarize(
                [
                    row
                    for row in rows_by_condition[condition]
                    if row["calculator_id"] == calculator_id
                ]
            )
            for condition in CONDITIONS
        }

    lookup = {
        (row["condition"], row["base_sample_id"]): row for row in case_rows
    }
    sample_ids = [target["base_sample_id"] for target in targets]
    metric_specs: list[tuple[str, str, Callable[[dict[str, Any]], bool]]] = [
        ("revision_correct", "higher", lambda row: bool(row["revision_correct"])),
        ("switch_success", "higher", lambda row: bool(row["switch_success"])),
        (
            "stale_original",
            "lower",
            lambda row: bool(row["stale_original"]),
        ),
        (
            "revision_action3_correct",
            "higher",
            lambda row: bool(row["revision_action3_correct"]),
        ),
        (
            "revision_final_and_pair_f1_ge_80",
            "higher",
            lambda row: bool(row["revision_final_and_pair_f1_ge_80"]),
        ),
    ]
    comparisons = []
    p_values = []
    for index, (metric, direction, getter) in enumerate(metric_specs):
        candidate = [
            getter(lookup[("krpo_ruleshift", sample_id)])
            for sample_id in sample_ids
        ]
        comparator = [
            getter(lookup[("outcome_ruleshift", sample_id)])
            for sample_id in sample_ids
        ]
        mcnemar = exact_mcnemar(candidate, comparator)
        result = {
            "metric": metric,
            "preferred_direction": direction,
            "krpo_rate": sum(candidate) / len(candidate),
            "outcome_rate": sum(comparator) / len(comparator),
            "krpo_minus_outcome": (
                sum(candidate) - sum(comparator)
            ) / len(candidate),
            "paired_bootstrap_95_ci": paired_bootstrap(
                [float(value) for value in candidate],
                [float(value) for value in comparator],
                BOOTSTRAP_SEED + index,
            ),
            "mcnemar": mcnemar,
        }
        comparisons.append(result)
        p_values.append(float(mcnemar["p_value_raw"]))
    for result, adjusted in zip(comparisons, holm(p_values)):
        result["mcnemar"]["p_value_holm"] = adjusted

    process_candidate = [
        float(lookup[("krpo_ruleshift", sample_id)]["revision_pair_f1"])
        for sample_id in sample_ids
    ]
    process_comparator = [
        float(lookup[("outcome_ruleshift", sample_id)]["revision_pair_f1"])
        for sample_id in sample_ids
    ]
    process_comparison = {
        "metric": "revision_canonical_pair_f1",
        "krpo_mean": sum(process_candidate) / len(process_candidate),
        "outcome_mean": sum(process_comparator) / len(process_comparator),
        "krpo_minus_outcome": (
            sum(process_candidate) - sum(process_comparator)
        )
        / len(process_candidate),
        "paired_bootstrap_95_ci": paired_bootstrap(
            process_candidate,
            process_comparator,
            BOOTSTRAP_SEED + 100,
        ),
    }
    output = {
        "status": "PASS",
        "n_paired_cases": len(targets),
        "n_prompts_per_model": len(targets) * 2,
        "primary_endpoint": "revision_correct",
        "summary": summary,
        "by_calculator": by_calculator,
        "krpo_vs_outcome_binary": comparisons,
        "krpo_vs_outcome_process": process_comparison,
        "interpretation_boundary": (
            "Synthetic controlled context-adherence stress test; revised "
            "weights are not clinical recommendations."
        ),
    }
    output_root = args.output_root.resolve()
    write_json(output_root / "summary.json", output)
    write_jsonl(output_root / "case_metrics.jsonl", case_rows)

    lines = [
        "# Task 10 MedCalc rule-revision results",
        "",
        f"Paired cases: **n={len(targets)}**; two prompts per case and model.",
        "",
        "| Condition | Control acc. | Revised-rule acc. | Stale-answer rate | Switch success | Revised pair-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = summary[condition]
        lines.append(
            f"| {condition} | {100*row['control_correct']:.1f}% | "
            f"{100*row['revision_correct']:.1f}% | "
            f"{100*row['stale_original']:.1f}% | "
            f"{100*row['switch_success']:.1f}% | "
            f"{100*row['revision_pair_f1']:.1f}% |"
        )
    lines.extend(["", "## K-RPO versus outcome-only", ""])
    for result in comparisons:
        low, high = result["paired_bootstrap_95_ci"]
        lines.append(
            f"- {result['metric']}: {100*result['krpo_minus_outcome']:+.1f} pp "
            f"(95% CI [{100*low:+.1f}, {100*high:+.1f}]; "
            f"Holm p={result['mcnemar']['p_value_holm']:.4g})."
        )
    low, high = process_comparison["paired_bootstrap_95_ci"]
    lines.append(
        f"- revision_canonical_pair_f1: "
        f"{100*process_comparison['krpo_minus_outcome']:+.1f} pp "
        f"(95% CI [{100*low:+.1f}, {100*high:+.1f}])."
    )
    lines.extend(
        [
            "",
            "The altered weights are synthetic and test contextual rule",
            "execution; they are not proposed as clinically valid updates.",
        ]
    )
    (output_root / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
