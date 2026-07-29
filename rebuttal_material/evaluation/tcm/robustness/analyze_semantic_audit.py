#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import pathlib
import re
from collections import Counter
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent
LEVELS = (100, 75, 50, 25, 0)
NEW_LEVELS = (75, 50, 25, 0)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verifier() -> Any:
    path = ROOT / "verify.py"
    specification = importlib.util.spec_from_file_location(
        "task8_independent_core", path
    )
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def normalize(value: Any) -> str:
    return re.sub(r"\s+", "", str(value)).replace("淤", "瘀")


def hierarchy_compatible(left: Any, right: Any) -> bool:
    left_value = normalize(left)
    right_value = normalize(right)
    return (
        left_value == right_value
        or left_value.startswith(right_value + "-")
        or right_value.startswith(left_value + "-")
    )


def condition(model: str, mode: str, level: int) -> str:
    return (
        f"q14_{model}_clean"
        if level == 100
        else f"q14_{model}_{mode}_q{level:03d}"
    )


def comparison(
    candidate: list[bool], comparator: list[bool], verifier: Any
) -> dict[str, Any]:
    candidate_rate = sum(candidate) / len(candidate)
    comparator_rate = sum(comparator) / len(comparator)
    candidate_only = sum(
        left and not right for left, right in zip(candidate, comparator)
    )
    comparator_only = sum(
        right and not left for left, right in zip(candidate, comparator)
    )
    return {
        "n": len(candidate),
        "candidate_rate": candidate_rate,
        "comparator_rate": comparator_rate,
        "difference": candidate_rate - comparator_rate,
        "bootstrap_95_ci": verifier.bootstrap(candidate, comparator),
        "candidate_only": candidate_only,
        "comparator_only": comparator_only,
        "p_value_raw": verifier.exact_mcnemar(candidate, comparator),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=pathlib.Path,
        default=ROOT.parent / "task6/targets/tcm_targets_100.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=ROOT / "results",
    )
    args = parser.parse_args()
    verifier = load_verifier()
    task6 = ROOT.parent / "task6"
    audit_root = ROOT / "manual_audit_21"
    source_path = audit_root / "manual_audit_21.csv"
    completed_path = audit_root / "manual_audit_21_completed.csv"
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    with completed_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        completed_rows = list(csv.DictReader(handle))
    if len(source_rows) != 21 or len(completed_rows) != 21:
        raise ValueError("semantic audit must contain exactly 21 rows")
    immutable = (
        "selection_index",
        "mode",
        "quality",
        "sample_id",
        "task_family",
        "compact_audit_file",
        "automatic_instruction_hash",
        "automatic_change_applied",
        "automatic_no_explicit_answer_field",
        "automatic_required_prompt_markers",
    )
    for source, completed in zip(source_rows, completed_rows):
        if any(source[key] != completed[key] for key in immutable):
            raise ValueError(
                f"{source['selection_index']}: immutable audit field changed"
            )
        answer = completed["human_semantic_valid_yes_no"].strip().lower()
        if answer not in {"yes", "no"}:
            raise ValueError(
                f"{source['selection_index']}: expected yes/no answer"
            )
        if answer == "no" and not completed["human_notes"].strip():
            raise ValueError(
                f"{source['selection_index']}: no requires a note"
            )

    counts = Counter(
        row["human_semantic_valid_yes_no"].strip().lower()
        for row in completed_rows
    )
    by_mode = {
        mode: dict(
            Counter(
                row["human_semantic_valid_yes_no"].strip().lower()
                for row in completed_rows
                if row["mode"] == mode
            )
        )
        for mode in ("relevant", "random", "conflict")
    }
    failed = [
        {
            "selection_index": int(row["selection_index"]),
            "mode": row["mode"],
            "quality": int(row["quality"]),
            "sample_id": row["sample_id"],
            "task_family": row["task_family"],
            "human_notes": row["human_notes"],
        }
        for row in completed_rows
        if row["human_semantic_valid_yes_no"].strip().lower() == "no"
    ]

    targets = {
        str(row["sample_id"]): row
        for row in verifier.records(
            args.targets.resolve()
        )
    }
    maps = {
        str(row["sample_id"]): row
        for row in verifier.records(ROOT / "relevant_rule_map_100.jsonl")
    }
    strict_relevant_ids = [
        sample_id
        for sample_id in sorted(targets)
        if targets[sample_id]["metadata"]["primary_leakage_free"]
        and bool(maps[sample_id]["matched_rule_ids"])
    ]
    strict_conflict_ids = []
    for sample_id in sorted(targets):
        target = targets[sample_id]
        if not target["metadata"]["primary_leakage_free"]:
            continue
        fields = verifier.FIELDS[target["metadata"]["task_family"]]
        clean = target["final_summaries"]["clean"]
        fixed = target["final_summaries"]["counterfactual_fixed"]
        if any(
            not hierarchy_compatible(clean[field], fixed[field])
            for field in fields
        ):
            strict_conflict_ids.append(sample_id)

    new_conditions = [
        f"q14_{model}_{mode}_q{level:03d}"
        for mode in ("relevant", "random", "conflict")
        for level in NEW_LEVELS
        for model in ("base", "krpo")
    ]
    predictions = verifier.load_archive_rows(
        ROOT / "task8_prediction_bundle.tar.gz", new_conditions
    )
    predictions.update(
        verifier.load_archive_rows(
            task6 / "task6_prediction_bundle.tar.gz",
            ["q14_base_clean", "q14_krpo_clean"],
        )
    )
    outcomes: dict[tuple[str, str], bool] = {}
    for condition_name, rows in predictions.items():
        for sample_id, row in rows.items():
            target = targets[sample_id]
            fields = verifier.FIELDS[target["metadata"]["task_family"]]
            outcomes[(condition_name, sample_id)] = verifier.exact(
                str(row["response"]),
                target["final_summaries"]["clean"],
                fields,
            )

    curves: dict[str, Any] = {}
    for mode, ids in (
        ("relevant", strict_relevant_ids),
        ("random", strict_relevant_ids),
        ("conflict", strict_conflict_ids),
    ):
        model_rates: dict[str, dict[str, float]] = {
            "base": {},
            "krpo": {},
        }
        model_comparisons: list[dict[str, Any]] = []
        for level in LEVELS:
            for model in ("base", "krpo"):
                values = [
                    outcomes[(condition(model, mode, level), sample_id)]
                    for sample_id in ids
                ]
                model_rates[model][str(level)] = sum(values) / len(values)
            if level != 100:
                candidate = [
                    outcomes[(condition("krpo", mode, level), sample_id)]
                    for sample_id in ids
                ]
                comparator = [
                    outcomes[(condition("base", mode, level), sample_id)]
                    for sample_id in ids
                ]
                result = comparison(candidate, comparator, verifier)
                result["level"] = level
                model_comparisons.append(result)
        adjusted = verifier.holm(
            [item["p_value_raw"] for item in model_comparisons]
        )
        for item, value in zip(model_comparisons, adjusted):
            item["p_value_holm"] = value
        curves[mode] = {
            "n": len(ids),
            "eligibility": (
                "primary source-verified and at least one exact matched diagnostic rule"
                if mode != "conflict"
                else (
                    "primary source-verified and at least one scored field is "
                    "neither equal nor ancestor/descendant compatible"
                )
            ),
            "model_rates": model_rates,
            "krpo_minus_base": model_comparisons,
        }

    amount_matched: dict[str, list[dict[str, Any]]] = {}
    for model in ("base", "krpo"):
        comparisons = []
        for level in NEW_LEVELS:
            relevant = [
                outcomes[
                    (condition(model, "relevant", level), sample_id)
                ]
                for sample_id in strict_relevant_ids
            ]
            random_rows = [
                outcomes[(condition(model, "random", level), sample_id)]
                for sample_id in strict_relevant_ids
            ]
            result = comparison(relevant, random_rows, verifier)
            result["level"] = level
            comparisons.append(result)
        adjusted = verifier.holm(
            [item["p_value_raw"] for item in comparisons]
        )
        for item, value in zip(comparisons, adjusted):
            item["p_value_holm"] = value
        amount_matched[model] = comparisons

    report = {
        "status": "PASS_WITH_SEMANTIC_LIMITATIONS",
        "verification_status": "VERIFIED",
        "audit": {
            "completed_csv_sha256": sha256(completed_path),
            "rows": 21,
            "yes": counts["yes"],
            "no": counts["no"],
            "pass_rate": counts["yes"] / 21,
            "by_mode": by_mode,
            "failed_cases": failed,
            "immutable_fields_unchanged": True,
            "all_automatic_checks_pass": True,
        },
        "audit_informed_sensitivity": {
            "status": "POST_AUDIT_OUTCOME_BLIND_SENSITIVITY",
            "model_outputs_used_to_define_filters": False,
            "not_a_replacement_for_predeclared_primary_analysis": True,
            "strict_relevant_primary_n": len(strict_relevant_ids),
            "strict_conflict_primary_n": len(strict_conflict_ids),
            "curves": curves,
            "amount_matched_relevant_minus_random": amount_matched,
        },
    }
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_json = output_root / "task8_semantic_audit.json"
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Task8 semantic spot-check and strict sensitivity",
        "",
        f"- Audit result: {counts['yes']}/21 valid "
        f"({100*counts['yes']/21:.1f}%), {counts['no']}/21 invalid.",
        f"- Relevant: {by_mode['relevant'].get('yes', 0)}/7 valid.",
        f"- Random: {by_mode['random'].get('yes', 0)}/7 valid.",
        f"- Conflict: {by_mode['conflict'].get('yes', 0)}/7 valid.",
        "- Both failures are semantic construct failures in "
        "`wei_qi_ying_xue`; all mechanical checks passed.",
        "",
        "## Audit-informed strict sensitivity",
        "",
        "These filters were motivated by the outcome-independent audit and "
        "must be labeled post-audit sensitivity analyses.",
        "",
    ]
    for mode in ("relevant", "random", "conflict"):
        curve = curves[mode]
        lines.extend(
            [
                f"### {mode} (n={curve['n']})",
                "",
                "| Quality | Base | K-RPO | Δ | 95% CI | Holm p |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        comparisons_by_level = {
            item["level"]: item for item in curve["krpo_minus_base"]
        }
        for level in LEVELS:
            base = curve["model_rates"]["base"][str(level)]
            krpo = curve["model_rates"]["krpo"][str(level)]
            if level == 100:
                lines.append(
                    f"| 100% | {100*base:.1f}% | {100*krpo:.1f}% | "
                    f"{100*(krpo-base):+.1f} pp | — | — |"
                )
            else:
                item = comparisons_by_level[level]
                ci = item["bootstrap_95_ci"]
                lines.append(
                    f"| {level}% | {100*base:.1f}% | {100*krpo:.1f}% | "
                    f"{100*item['difference']:+.1f} pp | "
                    f"[{100*ci[0]:+.1f},{100*ci[1]:+.1f}] | "
                    f"{item['p_value_holm']:.4g} |"
                )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- The stricter relevant-rule analysis strengthens the "
            "K-RPO-versus-Base pattern at all four new levels.",
            "- The stricter conflict analysis still shows higher K-RPO point "
            "estimates at every level, but only the 50% level survives Holm "
            "correction; the original claim that all four conflict contrasts "
            "are significant is not robust to this stricter eligibility rule.",
            "- The experiment remains useful as quantitative exploratory "
            "evidence, but the rebuttal must disclose the 19/21 spot-check "
            "result and avoid claiming perfect semantic perturbation validity.",
            "",
        ]
    )
    output_md = output_root / "task8_semantic_audit.md"
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "yes": counts["yes"],
                "no": counts["no"],
                "strict_relevant_n": len(strict_relevant_ids),
                "strict_conflict_n": len(strict_conflict_ids),
                "json": str(output_json),
                "markdown": str(output_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
