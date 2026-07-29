#!/usr/bin/env python3
"""Reward-independent full-100 evaluation of 32B Base versus 32B K-RPO."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import random
import statistics
from typing import Any


CONDITIONS = ("q32_base_clean", "q32_krpo_clean")
BOOTSTRAP_SEED = 20260728
BOOTSTRAP_SAMPLES = 10_000


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


def load_task7(path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location("frozen_task7", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen evaluator {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap_continuous(
    candidate: list[float], comparator: list[float]
) -> list[float]:
    if len(candidate) != len(comparator) or not candidate:
        raise ValueError("continuous bootstrap inputs are invalid")
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(candidate)
    differences = []
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = [rng.randrange(n) for _ in range(n)]
        differences.append(
            statistics.mean(candidate[index] for index in indices)
            - statistics.mean(comparator[index] for index in indices)
        )
    return [
        percentile(differences, 0.025),
        percentile(differences, 0.975),
    ]


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.1f}%"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Task12: independent full-100 Qwen3-32B evaluation",
        "",
        "All metrics use the frozen Task7 exact evaluator, not the training reward.",
        "",
        "| Condition | Final joint | Action-step exact | Action complete | Complete trajectory |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        result = summary["conditions"][condition]
        lines.append(
            "| "
            + " | ".join(
                (
                    condition,
                    fmt(result["final_joint"]["rate"]),
                    fmt(result["action_step_exact"]),
                    fmt(result["action_completeness"]),
                    fmt(result["complete_trajectory"]["rate"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Paired K-RPO minus Base comparisons",
            "",
            "| Endpoint | Difference | 95% bootstrap CI | raw p | Holm p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for result in summary["binary_comparisons"]:
        low, high = result["bootstrap_95_ci"]
        lines.append(
            f"| {result['endpoint']} | {fmt(result['difference'])} | "
            f"[{fmt(low)}, {fmt(high)}] | {result['p_value_raw']:.4g} | "
            f"{result['p_value_holm']:.4g} |"
        )
    continuous = summary["continuous_comparison"]
    low, high = continuous["bootstrap_95_ci"]
    lines.extend(
        [
            "",
            f"Action-step exact difference: {fmt(continuous['difference'])} "
            f"(paired bootstrap 95% CI [{fmt(low)}, {fmt(high)}]).",
            "",
            "Complete trajectory follows the frozen Task7/Task9 definition: "
            "every required categorical action step is exact. "
            "`final_and_action_joint` is retained separately as a stricter "
            "secondary diagnostic.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task12-root", type=pathlib.Path, required=True)
    parser.add_argument("--task7-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()

    root = args.task12_root.resolve()
    task7_root = args.task7_root.resolve()
    output_root = args.output_root.resolve()
    audit = json.loads(
        (root / "server_runs/audits/prediction_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if audit.get("status") != "PASS":
        raise ValueError("Task12 prediction audit is not PASS")
    task7 = load_task7(task7_root / "evaluate_task7.py")
    normalizer = task7.LabelNormalizer(
        json.loads(
            (task7_root / "normalization_map.json").read_text(encoding="utf-8")
        )
    )
    targets = {
        row["sample_id"]: row
        for row in read_jsonl(root / "targets/tcm_targets_full100.jsonl")
    }
    if len(targets) != 100:
        raise ValueError("expected 100 targets")

    case_rows: list[dict[str, Any]] = []
    by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        predictions = {
            row["sample_id"]: row
            for row in read_jsonl(
                root
                / f"server_runs/predictions/{condition}/predictions.jsonl"
            )
        }
        if set(predictions) != set(targets):
            raise ValueError(f"{condition}: prediction IDs do not match targets")
        rows_for_condition: dict[str, dict[str, Any]] = {}
        for sample_id in sorted(targets):
            target = targets[sample_id]
            family = target["metadata"]["task_family"]
            fields = task7.FAMILY_FIELDS[family]
            response = predictions[sample_id]["response"]
            final = task7.evaluate_final(
                response,
                target["final_summary"],
                fields,
                normalizer,
            )
            actions = task7.evaluate_actions(
                response,
                target["clean_output"],
                normalizer,
            )
            complete = bool(actions["action_trajectory_exact"])
            final_and_action_joint = bool(final["joint_exact"] and complete)
            row = {
                "condition": condition,
                "sample_id": sample_id,
                "metadata": target["metadata"],
                "truth": final,
                "truth_actions": actions,
                "complete_trajectory_exact": complete,
                "final_and_action_joint_exact": final_and_action_joint,
            }
            rows_for_condition[sample_id] = row
            case_rows.append(row)
        by_condition[condition] = rows_for_condition

    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [by_condition[condition][sample_id] for sample_id in sorted(targets)]
        frozen = task7.summarize_condition(rows, "truth")
        conditions[condition] = {
            "final_joint": frozen["joint_exact"],
            "schema_valid": frozen["schema_valid"],
            "field_micro_exact": frozen["field_micro_exact"],
            "family_macro_joint_exact": frozen["family_macro_joint_exact"],
            "by_family": frozen["by_family"],
            "action_step_exact": frozen["actions"]["action_step_exact"],
            "action_field_exact": frozen["actions"]["action_field_exact"],
            "action_completeness": frozen["actions"]["action_completeness"],
            "action_trajectory": frozen["actions"]["trajectory_exact"],
            "complete_trajectory": task7.summarize_binary(
                [bool(row["complete_trajectory_exact"]) for row in rows]
            ),
            "final_and_action_joint": task7.summarize_binary(
                [bool(row["final_and_action_joint_exact"]) for row in rows]
            ),
        }

    ids = sorted(targets)
    binary_specs = [
        (
            "final_joint_exact",
            lambda row: bool(row["truth"]["joint_exact"]),
        ),
        (
            "complete_trajectory_exact",
            lambda row: bool(row["complete_trajectory_exact"]),
        ),
    ]
    binary_results = []
    for endpoint, getter in binary_specs:
        candidate = [
            getter(by_condition["q32_krpo_clean"][sample_id])
            for sample_id in ids
        ]
        comparator = [
            getter(by_condition["q32_base_clean"][sample_id])
            for sample_id in ids
        ]
        mcnemar = task7.exact_mcnemar(candidate, comparator)
        binary_results.append(
            {
                "endpoint": endpoint,
                "n": len(ids),
                "candidate": "q32_krpo_clean",
                "comparator": "q32_base_clean",
                "candidate_rate": task7.accuracy(candidate),
                "comparator_rate": task7.accuracy(comparator),
                "difference": (
                    task7.accuracy(candidate) - task7.accuracy(comparator)
                ),
                "bootstrap_95_ci": task7.paired_bootstrap_difference(
                    candidate,
                    comparator,
                    samples=BOOTSTRAP_SAMPLES,
                    seed=BOOTSTRAP_SEED,
                ),
                **mcnemar,
            }
        )
    adjusted = task7.holm_adjust(
        [float(result["p_value_raw"]) for result in binary_results]
    )
    for result, p_value in zip(binary_results, adjusted):
        result["p_value_holm"] = p_value

    candidate_steps = [
        float(
            by_condition["q32_krpo_clean"][sample_id]["truth_actions"][
                "action_step_exact"
            ]
        )
        for sample_id in ids
    ]
    comparator_steps = [
        float(
            by_condition["q32_base_clean"][sample_id]["truth_actions"][
                "action_step_exact"
            ]
        )
        for sample_id in ids
    ]
    continuous = {
        "endpoint": "per-case action_step_exact",
        "n": len(ids),
        "candidate_mean": statistics.mean(candidate_steps),
        "comparator_mean": statistics.mean(comparator_steps),
        "difference": (
            statistics.mean(candidate_steps)
            - statistics.mean(comparator_steps)
        ),
        "bootstrap_95_ci": paired_bootstrap_continuous(
            candidate_steps, comparator_steps
        ),
    }
    summary = {
        "status": "PASS",
        "analysis_population": "all 100 patient-task instances",
        "evaluator": {
            "description": (
                "frozen Task7 typed-decision and action exact-match evaluator"
            ),
            "independent_of": [
                "training reward implementation",
                "embedding similarity",
            ],
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "multiplicity": (
                "Holm adjustment across the two pre-specified binary endpoints"
            ),
        },
        "conditions": conditions,
        "binary_comparisons": binary_results,
        "continuous_comparison": continuous,
    }
    write_jsonl(output_root / "case_metrics.jsonl", case_rows)
    write_json(output_root / "summary.json", summary)
    (output_root / "summary.md").write_text(
        markdown(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
