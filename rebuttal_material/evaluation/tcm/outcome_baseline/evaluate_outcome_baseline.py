#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import random
import statistics
import tarfile
from typing import Any


BOOTSTRAP_SEED = 20260727
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
    spec = importlib.util.spec_from_file_location("task7_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen evaluator {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bundle_predictions(path: pathlib.Path) -> list[dict[str, Any]]:
    member = (
        "server_runs/predictions/q14_outcome_clean/predictions.jsonl"
    )
    with tarfile.open(path, "r:gz") as archive:
        handle = archive.extractfile(member)
        if handle is None:
            raise FileNotFoundError(f"{path}: missing {member}")
        return [
            json.loads(line)
            for line in handle.read().decode("utf-8").splitlines()
            if line.strip()
        ]


def percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def paired_bootstrap_continuous(
    candidate: list[float], comparator: list[float]
) -> list[float]:
    if len(candidate) != len(comparator) or not candidate:
        raise ValueError("continuous paired bootstrap inputs are invalid")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task9-root", type=pathlib.Path, required=True)
    parser.add_argument("--task7-root", type=pathlib.Path, required=True)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--targets",
        type=pathlib.Path,
        default=None,
        help="Optional audited target file.",
    )
    parser.add_argument(
        "--task7-case-metrics",
        type=pathlib.Path,
        default=None,
        help="Optional audited case metrics.",
    )
    args = parser.parse_args()

    task9 = args.task9_root.resolve()
    task7_root = args.task7_root.resolve()
    task7 = load_task7(task7_root / "evaluate.py")
    normalizer = task7.LabelNormalizer(
        json.loads(
            (task9 / "normalization_map.json").read_text(encoding="utf-8")
        )
    )
    targets_path = (
        args.targets.resolve()
        if args.targets is not None
        else task9 / "targets/tcm_targets_100.jsonl"
    )
    targets = {
        row["sample_id"]: row
        for row in read_jsonl(targets_path)
    }
    outcome_predictions = {
        row["sample_id"]: row for row in bundle_predictions(args.bundle)
    }
    if set(outcome_predictions) != set(targets):
        raise ValueError("Task 9 prediction IDs do not match frozen targets")
    if not all(
        row.get("status") == "completed" for row in outcome_predictions.values()
    ):
        raise ValueError("Task 9 bundle contains incomplete predictions")

    task7_case_metrics = (
        args.task7_case_metrics.resolve()
        if args.task7_case_metrics is not None
        else task7_root / "results/case_metrics.jsonl"
    )
    task7_rows = read_jsonl(task7_case_metrics)
    prior = {
        (row["condition"], row["sample_id"]): row
        for row in task7_rows
        if row["condition"] in {"q14_base_clean", "q14_krpo_clean"}
    }
    if len(prior) != 200:
        raise ValueError("expected 200 Base/K-RPO Task 7 case rows")

    outcome_rows = []
    for sample_id in sorted(targets):
        target = targets[sample_id]
        family = target["metadata"]["task_family"]
        fields = task7.FAMILY_FIELDS[family]
        response = outcome_predictions[sample_id]["response"]
        final = task7.evaluate_final(
            response,
            target["final_summaries"]["clean"],
            fields,
            normalizer,
        )
        actions = task7.evaluate_actions(
            response, target["clean_output"], normalizer
        )
        outcome_rows.append(
            {
                "condition": "q14_outcome_clean",
                "sample_id": sample_id,
                "metadata": target["metadata"],
                "truth": final,
                "truth_actions": actions,
            }
        )

    combined = []
    for sample_id in sorted(targets):
        combined.extend(
            [
                prior[("q14_base_clean", sample_id)],
                next(
                    row
                    for row in outcome_rows
                    if row["sample_id"] == sample_id
                ),
                prior[("q14_krpo_clean", sample_id)],
            ]
        )
    primary_ids = [
        sample_id
        for sample_id in sorted(targets)
        if targets[sample_id]["metadata"]["primary_leakage_free"]
    ]
    by_condition = {
        condition: [
            row
            for row in combined
            if row["condition"] == condition
            and row["sample_id"] in set(primary_ids)
        ]
        for condition in (
            "q14_base_clean",
            "q14_outcome_clean",
            "q14_krpo_clean",
        )
    }
    summaries = {
        condition: task7.summarize_condition(rows, "truth")
        for condition, rows in by_condition.items()
    }
    row_lookup = {
        (row["condition"], row["sample_id"]): row for row in combined
    }

    binary_specs = [
        (
            "krpo_minus_outcome_final_joint",
            "q14_krpo_clean",
            "q14_outcome_clean",
            lambda row: bool(row["truth"]["joint_exact"]),
        ),
        (
            "krpo_minus_outcome_action_trajectory",
            "q14_krpo_clean",
            "q14_outcome_clean",
            lambda row: bool(row["truth_actions"]["action_trajectory_exact"]),
        ),
        (
            "outcome_minus_base_final_joint",
            "q14_outcome_clean",
            "q14_base_clean",
            lambda row: bool(row["truth"]["joint_exact"]),
        ),
        (
            "outcome_minus_base_action_trajectory",
            "q14_outcome_clean",
            "q14_base_clean",
            lambda row: bool(row["truth_actions"]["action_trajectory_exact"]),
        ),
    ]
    comparisons = []
    p_values = []
    for name, candidate_condition, comparator_condition, getter in binary_specs:
        candidate = [
            getter(row_lookup[(candidate_condition, sample_id)])
            for sample_id in primary_ids
        ]
        comparator = [
            getter(row_lookup[(comparator_condition, sample_id)])
            for sample_id in primary_ids
        ]
        mcnemar = task7.exact_mcnemar(candidate, comparator)
        result = {
            "name": name,
            "candidate_condition": candidate_condition,
            "comparator_condition": comparator_condition,
            "candidate_rate": sum(candidate) / len(candidate),
            "comparator_rate": sum(comparator) / len(comparator),
            "difference": (
                sum(candidate) - sum(comparator)
            ) / len(candidate),
            "bootstrap_95_ci": task7.paired_bootstrap_difference(
                candidate,
                comparator,
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
            ),
            "mcnemar": mcnemar,
        }
        comparisons.append(result)
        p_values.append(float(mcnemar["p_value_raw"]))
    adjusted = task7.holm_adjust(p_values)
    for result, value in zip(comparisons, adjusted):
        result["mcnemar"]["p_value_holm"] = value

    continuous = []
    for candidate_condition, comparator_condition in (
        ("q14_krpo_clean", "q14_outcome_clean"),
        ("q14_outcome_clean", "q14_base_clean"),
    ):
        candidate = [
            float(
                row_lookup[(candidate_condition, sample_id)]["truth_actions"][
                    "action_step_exact"
                ]
            )
            for sample_id in primary_ids
        ]
        comparator = [
            float(
                row_lookup[(comparator_condition, sample_id)]["truth_actions"][
                    "action_step_exact"
                ]
            )
            for sample_id in primary_ids
        ]
        continuous.append(
            {
                "name": (
                    f"{candidate_condition}_minus_{comparator_condition}"
                    "_action_step_exact"
                ),
                "candidate_mean": statistics.mean(candidate),
                "comparator_mean": statistics.mean(comparator),
                "difference": statistics.mean(candidate)
                - statistics.mean(comparator),
                "bootstrap_95_ci": paired_bootstrap_continuous(
                    candidate, comparator
                ),
            }
        )

    summary = {
        "status": "PASS",
        "population": "source-verified TCM cases",
        "n": len(primary_ids),
        "scoring": (
            "Frozen Task 7 normalized exact match; no training reward, "
            "embedding, fuzzy match, or free-text conclusion score."
        ),
        "conditions": summaries,
        "binary_comparisons": comparisons,
        "continuous_comparisons": continuous,
    }
    output_root = args.output_root.resolve()
    write_json(output_root / "summary.json", summary)
    write_jsonl(output_root / "case_metrics.jsonl", combined)

    lines = [
        "# Task matched TCM outcome-only comparison",
        "",
        f"Source-verified cases: **n={len(primary_ids)}**.",
        "",
        "| Condition | Truth joint | Action-step exact | Action complete |",
        "|---|---:|---:|---:|",
    ]
    for condition in (
        "q14_base_clean",
        "q14_outcome_clean",
        "q14_krpo_clean",
    ):
        value = summaries[condition]
        lines.append(
            f"| {condition} | "
            f"{100 * value['joint_exact']['rate']:.1f}% | "
            f"{100 * value['actions']['action_step_exact']:.1f}% | "
            f"{100 * value['actions']['action_completeness']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "All endpoints use the frozen reward-independent Task 7 evaluator.",
            "",
            "## Paired comparisons",
            "",
        ]
    )
    for result in comparisons:
        low, high = result["bootstrap_95_ci"]
        lines.append(
            f"- {result['name']}: {100 * result['difference']:+.1f} pp "
            f"(95% CI [{100 * low:+.1f}, {100 * high:+.1f}]; "
            f"Holm p={result['mcnemar']['p_value_holm']:.4g})."
        )
    for result in continuous:
        low, high = result["bootstrap_95_ci"]
        lines.append(
            f"- {result['name']}: {100 * result['difference']:+.1f} pp "
            f"(paired bootstrap 95% CI "
            f"[{100 * low:+.1f}, {100 * high:+.1f}])."
        )
    (output_root / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
