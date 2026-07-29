#!/usr/bin/env python3
"""Paired four-condition analysis for the MedCalc  experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
from collections import Counter
from typing import Any


CONDITIONS = (
    "base_norule",
    "base_rule",
    "outcome_grpo_rule",
    "krpo_rule",
)
COMPARISONS = (
    ("krpo_rule", "outcome_grpo_rule"),
    ("krpo_rule", "base_rule"),
    ("base_rule", "base_norule"),
)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def accuracy(rows: list[dict[str, Any]]) -> float:
    return (
        sum(bool(row["final_official_correct"]) for row in rows) / len(rows)
        if rows
        else 0.0
    )


def metric_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "final_official_accuracy": accuracy(rows),
        "malformed_rate": sum(bool(row["malformed"]) for row in rows) / n,
        "calculator_rule_exact": sum(
            bool(row["calculator_rule_exact"]) for row in rows
        )
        / n,
        "entity_exact": sum(bool(row["entity_exact"]) for row in rows) / n,
        "trace_text_exact": sum(bool(row["trace_text_exact"]) for row in rows)
        / n,
    }


def exact_mcnemar_p(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(b, c) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1 - fraction)
        + sorted_values[upper] * fraction
    )


def paired_comparison(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    ids = sorted(left)
    left_values = [bool(left[item]["final_official_correct"]) for item in ids]
    right_values = [bool(right[item]["final_official_correct"]) for item in ids]
    left_wins = sum(a and not b for a, b in zip(left_values, right_values))
    right_wins = sum(b and not a for a, b in zip(left_values, right_values))
    observed = (
        sum(left_values) / len(ids) - sum(right_values) / len(ids)
    )
    generator = random.Random(seed)
    deltas = []
    for _ in range(bootstrap_samples):
        sample = [generator.randrange(len(ids)) for _ in ids]
        deltas.append(
            sum(left_values[index] for index in sample) / len(ids)
            - sum(right_values[index] for index in sample) / len(ids)
        )
    deltas.sort()
    return {
        "n": len(ids),
        "left_accuracy": sum(left_values) / len(ids),
        "right_accuracy": sum(right_values) / len(ids),
        "accuracy_difference": observed,
        "accuracy_difference_percentage_points": observed * 100,
        "paired_bootstrap_95ci": [
            percentile(deltas, 0.025),
            percentile(deltas, 0.975),
        ],
        "left_correct_right_wrong": left_wins,
        "left_wrong_right_correct": right_wins,
        "exact_mcnemar_p": exact_mcnemar_p(left_wins, right_wins),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (total - rank) * p_values[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    root = args.evaluation_root.resolve()
    rows: dict[str, list[dict[str, Any]]] = {}
    maps: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        path = root / condition / "evaluation" / "detailed_results.jsonl"
        rows[condition] = read_jsonl(path)
        maps[condition] = {str(row["sample_id"]): row for row in rows[condition]}
        if len(rows[condition]) != 380 or len(maps[condition]) != 380:
            raise ValueError(f"{condition}: expected 380 unique paired results")
    reference_ids = set(maps[CONDITIONS[0]])
    if any(set(maps[condition]) != reference_ids for condition in CONDITIONS[1:]):
        raise ValueError("The four conditions do not contain identical sample IDs")

    invariant_fields = (
        "calculator_id",
        "calculator_name",
        "category",
        "evaluation_split",
        "ground_truth_answer",
    )
    for sample_id in reference_ids:
        baseline = maps[CONDITIONS[0]][sample_id]
        for condition in CONDITIONS[1:]:
            candidate = maps[condition][sample_id]
            for field in invariant_fields:
                if candidate[field] != baseline[field]:
                    raise ValueError(
                        f"{sample_id}: field {field} differs across conditions"
                    )

    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "overall": {"all": rows[CONDITIONS[0]]},
        "evaluation_split": {},
        "category": {},
    }
    for group_field in ("evaluation_split", "category"):
        values = sorted(
            {str(row[group_field]) for row in rows[CONDITIONS[0]]}
        )
        groups[group_field] = {
            value: [
                row
                for row in rows[CONDITIONS[0]]
                if str(row[group_field]) == value
            ]
            for value in values
        }

    table: dict[str, Any] = {}
    for family, family_groups in groups.items():
        table[family] = {}
        for group_name, baseline_group in family_groups.items():
            ids = {str(row["sample_id"]) for row in baseline_group}
            table[family][group_name] = {
                condition: metric_row(
                    [row for row in rows[condition] if str(row["sample_id"]) in ids]
                )
                for condition in CONDITIONS
            }

    pairwise: dict[str, Any] = {}
    for index, (left, right) in enumerate(COMPARISONS):
        name = f"{left}_vs_{right}"
        pairwise[name] = paired_comparison(
            maps[left],
            maps[right],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + index,
        )
    adjusted = holm_adjust(
        {name: result["exact_mcnemar_p"] for name, result in pairwise.items()}
    )
    for name in pairwise:
        pairwise[name]["holm_adjusted_p"] = adjusted[name]

    output = {
        "status": "PASS",
        "conditions": list(CONDITIONS),
        "n": 380,
        "split_counts": Counter(
            row["evaluation_split"] for row in rows[CONDITIONS[0]]
        ),
        "category_counts": Counter(
            row["category"] for row in rows[CONDITIONS[0]]
        ),
        "descriptive_metrics": table,
        "primary_paired_comparisons": pairwise,
        "statistics": {
            "unit": "same test case paired across conditions",
            "primary_endpoint": "official final accuracy",
            "mcnemar": "two-sided exact binomial",
            "confidence_interval": (
                f"paired percentile bootstrap, {args.bootstrap_samples} samples"
            ),
            "multiplicity": "Holm adjustment across three predeclared comparisons",
            "seed": args.seed,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task5_comparison.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    paired_fields = [
        "sample_id",
        "evaluation_split",
        "category",
        *[f"{condition}_correct" for condition in CONDITIONS],
    ]
    with (args.output_dir / "paired_outcomes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=paired_fields)
        writer.writeheader()
        for sample_id in sorted(reference_ids):
            anchor = maps[CONDITIONS[0]][sample_id]
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "evaluation_split": anchor["evaluation_split"],
                    "category": anchor["category"],
                    **{
                        f"{condition}_correct": int(
                            bool(
                                maps[condition][sample_id][
                                    "final_official_correct"
                                ]
                            )
                        )
                        for condition in CONDITIONS
                    },
                }
            )

    overall = table["overall"]["all"]
    lines = [
        "#MedCalc comparison",
        "",
        "| Condition | Acc. | Malformed | Rule ID/name exact | Entity exact |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metric = overall[condition]
        lines.append(
            f"| {condition} | {metric['final_official_accuracy']:.3f} | "
            f"{metric['malformed_rate']:.3f} | "
            f"{metric['calculator_rule_exact']:.3f} | "
            f"{metric['entity_exact']:.3f} |"
        )
    lines.extend(["", "## Predeclared paired comparisons", ""])
    for name, result in pairwise.items():
        low, high = result["paired_bootstrap_95ci"]
        lines.append(
            f"- `{name}`: Δ={result['accuracy_difference_percentage_points']:.1f} "
            f"pp (95% paired bootstrap CI "
            f"[{low * 100:.1f}, {high * 100:.1f}]); exact McNemar "
            f"p={result['exact_mcnemar_p']:.4g}, Holm-adjusted "
            f"p={result['holm_adjusted_p']:.4g}."
        )
    (args.output_dir / "task5_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
