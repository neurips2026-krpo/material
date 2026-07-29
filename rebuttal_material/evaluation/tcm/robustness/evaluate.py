#!/usr/bin/env python3
"""Reward-independent evaluation of Task 8 knowledge-imperfection curves."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import tarfile
from typing import Any


LEVELS = (100, 75, 50, 25, 0)
NEW_LEVELS = (75, 50, 25, 0)
MODES = ("relevant", "random", "conflict")
MODE_ELIGIBILITY = {
    "relevant": "eligible_relevant_deletion",
    "random": "eligible_relevant_deletion",
    "conflict": "eligible_conflict",
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_module(path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location("task7_eval_frozen", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_task8_bundle(
    path: pathlib.Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    sidecar = pathlib.Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"missing checksum sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if sha256(path) != expected:
        raise ValueError("Task 8 prediction bundle checksum mismatch")
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        if any(
            not member.isfile()
            or pathlib.PurePosixPath(member.name).is_absolute()
            or ".." in pathlib.PurePosixPath(member.name).parts
            for member in members
        ):
            raise ValueError("unsafe Task 8 archive member")

        def payload(name: str) -> bytes:
            member = archive.getmember(name)
            handle = archive.extractfile(member)
            if handle is None:
                raise FileNotFoundError(name)
            return handle.read()

        manifest = json.loads(payload("task8_input_manifest.json"))
        audit = json.loads(
            payload("server_runs/audits/task8_prediction_manifest.json")
        )
        package = json.loads(
            payload("server_runs/audits/task8_bundle_manifest.json")
        )
        if manifest.get("status") != "PASS" or audit.get("status") != "PASS":
            raise ValueError("Task 8 manifest or prediction audit is not PASS")
        predictions: dict[str, list[dict[str, Any]]] = {}
        package_hashes = {
            item["relative_path"]: item["sha256"] for item in package["files"]
        }
        for input_condition in manifest["generated_input_files"]:
            for model in ("base", "krpo"):
                condition = f"q14_{model}_{input_condition}"
                name = (
                    f"server_runs/predictions/{condition}/predictions.jsonl"
                )
                raw = payload(name)
                observed = hashlib.sha256(raw).hexdigest()
                if package_hashes.get(name) != observed:
                    raise ValueError(f"{condition}: package hash mismatch")
                rows = [
                    json.loads(line)
                    for line in raw.decode("utf-8").splitlines()
                    if line.strip()
                ]
                if len(rows) != 100:
                    raise ValueError(f"{condition}: expected 100 rows")
                predictions[condition] = rows
    return predictions, manifest, audit


def condition_name(model: str, mode: str, level: int) -> str:
    if level == 100:
        return f"q14_{model}_clean"
    return f"q14_{model}_{mode}_q{level:03d}"


def metric_rows(
    predictions: dict[str, list[dict[str, Any]]],
    targets: dict[str, dict[str, Any]],
    evaluator: Any,
    normalizer: Any,
    *,
    conflict: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, values in predictions.items():
        for prediction in values:
            sample_id = str(prediction["sample_id"])
            target = targets[sample_id]
            metadata = target["metadata"]
            fields = evaluator.FAMILY_FIELDS[metadata["task_family"]]
            response = str(prediction["response"])
            row = {
                "sample_id": sample_id,
                "condition": condition,
                "metadata": metadata,
                "response_sha256": prediction["response_sha256"],
                "truth": evaluator.evaluate_final(
                    response,
                    target["final_summaries"]["clean"],
                    fields,
                    normalizer,
                ),
                "truth_actions": evaluator.evaluate_actions(
                    response, target["clean_output"], normalizer
                ),
                "adherence": None,
                "adherence_actions": None,
            }
            if conflict:
                row["adherence"] = evaluator.evaluate_final(
                    response,
                    target["final_summaries"]["counterfactual_fixed"],
                    fields,
                    normalizer,
                )
                row["adherence_actions"] = evaluator.evaluate_actions(
                    response,
                    target["counterfactual_fixed_output"],
                    normalizer,
                )
            rows.append(row)
    return rows


def binary_by_id(
    rows: list[dict[str, Any]], metric: str
) -> dict[str, bool]:
    return {
        row["sample_id"]: bool(row[metric]["joint_exact"]) for row in rows
    }


def paired_comparison(
    candidate: dict[str, bool],
    comparator: dict[str, bool],
    evaluator: Any,
    *,
    name: str,
) -> dict[str, Any]:
    ids = sorted(set(candidate) & set(comparator))
    if set(candidate) != set(comparator) or not ids:
        raise ValueError(f"{name}: comparison IDs are not exactly paired")
    left = [candidate[sample_id] for sample_id in ids]
    right = [comparator[sample_id] for sample_id in ids]
    return {
        "name": name,
        "n": len(ids),
        "candidate_rate": evaluator.accuracy(left),
        "comparator_rate": evaluator.accuracy(right),
        "difference": evaluator.accuracy(left) - evaluator.accuracy(right),
        "bootstrap_95_ci": evaluator.paired_bootstrap_difference(left, right),
        **evaluator.exact_mcnemar(left, right),
    }


def curve_auc(level_summaries: dict[str, Any]) -> float:
    ordered = sorted((int(level), value) for level, value in level_summaries.items())
    area = 0.0
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        area += (x1 - x0) * (
            y0["truth"]["joint_exact"]["rate"]
            + y1["truth"]["joint_exact"]["rate"]
        ) / 2
    return area / 100


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Task 8 knowledge-imperfection results",
        "",
        f"Status: **{summary['status']}**",
        "",
        "Primary tables below use the predeclared eligible set.",
        "",
    ]
    primary = summary["populations"]["primary"]
    for mode in MODES:
        lines.extend(
            [
                f"## {mode}",
                "",
                "| Quality | n | Base truth | K-RPO truth | Δ (pp) | 95% CI (pp) | Holm p |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        by_level = {
            item["level"]: item
            for item in primary["curves"][mode]["model_comparisons_truth"]
        }
        for level in LEVELS:
            base = primary["curves"][mode]["models"]["base"][str(level)]
            krpo = primary["curves"][mode]["models"]["krpo"][str(level)]
            if level == 100:
                comparison = {
                    "difference": (
                        krpo["truth"]["joint_exact"]["rate"]
                        - base["truth"]["joint_exact"]["rate"]
                    ),
                    "bootstrap_95_ci": [None, None],
                    "p_value_holm": None,
                }
            else:
                comparison = by_level[level]
            ci = comparison["bootstrap_95_ci"]
            ci_text = (
                "—" if ci[0] is None else f"[{100*ci[0]:.1f}, {100*ci[1]:.1f}]"
            )
            p_text = (
                "—"
                if comparison["p_value_holm"] is None
                else f"{comparison['p_value_holm']:.4g}"
            )
            lines.append(
                f"| {level}% | {base['truth']['joint_exact']['n']} | "
                f"{100*base['truth']['joint_exact']['rate']:.1f}% | "
                f"{100*krpo['truth']['joint_exact']['rate']:.1f}% | "
                f"{100*comparison['difference']:+.1f} | {ci_text} | {p_text} |"
            )
        lines.append("")
    controls = primary["amount_matched_relevant_minus_random_truth"]
    lines.extend(
        [
            "## Amount-matched relevant deletion minus random deletion",
            "",
            "| Model | Quality | Δ truth (pp) | 95% CI (pp) | Holm p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in ("base", "krpo"):
        for item in controls[model]:
            ci = item["bootstrap_95_ci"]
            lines.append(
                f"| {model} | {item['level']}% | "
                f"{100*item['difference']:+.1f} | "
                f"[{100*ci[0]:.1f}, {100*ci[1]:.1f}] | "
                f"{item['p_value_holm']:.4g} |"
            )
    lines.append("")
    lines.extend(
        [
            "Interpretation boundary: performance under missing knowledge is "
            "robustness evidence; following injected false knowledge is not "
            "clinical robustness and is reported separately as a safety boundary.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task8-bundle", type=pathlib.Path, required=True)
    parser.add_argument("--task6-bundle", type=pathlib.Path, required=True)
    parser.add_argument("--targets", type=pathlib.Path, required=True)
    parser.add_argument("--rule-map", type=pathlib.Path, required=True)
    parser.add_argument("--normalization", type=pathlib.Path, required=True)
    parser.add_argument("--task7-evaluator", type=pathlib.Path, required=True)
    parser.add_argument("--protocol", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()

    evaluator = load_module(args.task7_evaluator.resolve())
    normalizer = evaluator.LabelNormalizer(read_json(args.normalization))
    targets_list = read_jsonl(args.targets)
    targets = {str(row["sample_id"]): row for row in targets_list}
    maps = {str(row["sample_id"]): row for row in read_jsonl(args.rule_map)}
    if len(targets) != 100 or len(maps) != 100:
        raise ValueError("Task 8 requires 100 targets and 100 mapping rows")
    new_predictions, input_manifest, audit = load_task8_bundle(
        args.task8_bundle.resolve()
    )
    task6_predictions, _, task6_audit = evaluator.load_bundle(
        args.task6_bundle.resolve()
    )

    scored: dict[str, list[dict[str, Any]]] = {}
    for condition, predictions in new_predictions.items():
        scored[condition] = metric_rows(
            {condition: predictions},
            targets,
            evaluator,
            normalizer,
            conflict="_conflict_" in condition,
        )
    for condition in (
        "q14_base_clean",
        "q14_krpo_clean",
        "q14_base_nokg",
        "q14_base_cf",
        "q14_krpo_cf",
    ):
        scored[condition] = metric_rows(
            {condition: task6_predictions[condition]},
            targets,
            evaluator,
            normalizer,
            conflict=condition.endswith("_cf"),
        )

    populations: dict[str, Any] = {}
    for population, primary_only in (("primary", True), ("continuity", False)):
        population_result: dict[str, Any] = {"curves": {}, "anchors": {}}
        for mode in MODES:
            eligible_key = MODE_ELIGIBILITY[mode]
            ids = sorted(
                sample_id
                for sample_id, mapping in maps.items()
                if mapping[eligible_key]
                and (
                    not primary_only
                    or targets[sample_id]["metadata"]["primary_leakage_free"]
                )
            )
            models: dict[str, dict[str, Any]] = {"base": {}, "krpo": {}}
            truth_comparisons: list[dict[str, Any]] = []
            adherence_comparisons: list[dict[str, Any]] = []
            degradations: dict[str, list[dict[str, Any]]] = {
                "base": [],
                "krpo": [],
            }
            for model in ("base", "krpo"):
                clean_rows = [
                    row
                    for row in scored[f"q14_{model}_clean"]
                    if row["sample_id"] in ids
                ]
                clean_binary = binary_by_id(clean_rows, "truth")
                for level in LEVELS:
                    name = condition_name(model, mode, level)
                    rows = [
                        row for row in scored[name] if row["sample_id"] in ids
                    ]
                    models[model][str(level)] = {
                        "truth": evaluator.summarize_condition(rows, "truth")
                    }
                    if mode == "conflict" and level != 100:
                        models[model][str(level)]["adherence"] = (
                            evaluator.summarize_condition(rows, "adherence")
                        )
                    if level != 100:
                        degradation = paired_comparison(
                            binary_by_id(rows, "truth"),
                            clean_binary,
                            evaluator,
                            name=f"{model}_{mode}_q{level:03d}_minus_clean",
                        )
                        degradation["level"] = level
                        degradations[model].append(degradation)
            for level in NEW_LEVELS:
                base_rows = [
                    row
                    for row in scored[condition_name("base", mode, level)]
                    if row["sample_id"] in ids
                ]
                krpo_rows = [
                    row
                    for row in scored[condition_name("krpo", mode, level)]
                    if row["sample_id"] in ids
                ]
                comparison = paired_comparison(
                    binary_by_id(krpo_rows, "truth"),
                    binary_by_id(base_rows, "truth"),
                    evaluator,
                    name=f"krpo_minus_base_{mode}_truth_q{level:03d}",
                )
                comparison["level"] = level
                truth_comparisons.append(comparison)
                if mode == "conflict":
                    adherence = paired_comparison(
                        binary_by_id(krpo_rows, "adherence"),
                        binary_by_id(base_rows, "adherence"),
                        evaluator,
                        name=(
                            f"krpo_minus_base_{mode}_adherence_q{level:03d}"
                        ),
                    )
                    adherence["level"] = level
                    adherence_comparisons.append(adherence)
            for comparisons in (truth_comparisons, adherence_comparisons):
                adjusted = evaluator.holm_adjust(
                    [item["p_value_raw"] for item in comparisons]
                )
                for item, value in zip(comparisons, adjusted):
                    item["p_value_holm"] = value
            population_result["curves"][mode] = {
                "eligible_cases": len(ids),
                "models": models,
                "truth_auc": {
                    model: curve_auc(models[model]) for model in models
                },
                "model_comparisons_truth": truth_comparisons,
                "model_comparisons_adherence": adherence_comparisons,
                "degradation_from_clean_truth": degradations,
            }

        amount_matched: dict[str, list[dict[str, Any]]] = {}
        deletion_ids = sorted(
            sample_id
            for sample_id, mapping in maps.items()
            if mapping["eligible_relevant_deletion"]
            and (
                not primary_only
                or targets[sample_id]["metadata"]["primary_leakage_free"]
            )
        )
        for model in ("base", "krpo"):
            comparisons: list[dict[str, Any]] = []
            for level in NEW_LEVELS:
                relevant_rows = [
                    row
                    for row in scored[
                        condition_name(model, "relevant", level)
                    ]
                    if row["sample_id"] in deletion_ids
                ]
                random_rows = [
                    row
                    for row in scored[condition_name(model, "random", level)]
                    if row["sample_id"] in deletion_ids
                ]
                comparison = paired_comparison(
                    binary_by_id(relevant_rows, "truth"),
                    binary_by_id(random_rows, "truth"),
                    evaluator,
                    name=(
                        f"{model}_relevant_minus_random_truth_q{level:03d}"
                    ),
                )
                comparison["level"] = level
                comparisons.append(comparison)
            adjusted = evaluator.holm_adjust(
                [item["p_value_raw"] for item in comparisons]
            )
            for item, value in zip(comparisons, adjusted):
                item["p_value_holm"] = value
            amount_matched[model] = comparisons
        population_result[
            "amount_matched_relevant_minus_random_truth"
        ] = amount_matched

        population_ids = {
            sample_id
            for sample_id, target in targets.items()
            if not primary_only
            or target["metadata"]["primary_leakage_free"]
        }
        for condition, metric in (
            ("q14_base_nokg", "truth"),
            ("q14_base_cf", "truth"),
            ("q14_krpo_cf", "truth"),
            ("q14_base_cf", "adherence"),
            ("q14_krpo_cf", "adherence"),
        ):
            rows = [
                row for row in scored[condition] if row["sample_id"] in population_ids
            ]
            population_result["anchors"][f"{condition}:{metric}"] = (
                evaluator.summarize_condition(rows, metric)
            )
        populations[population] = population_result

    summary = {
        "status": "PASS",
        "evaluator": "Task 7 frozen exact evaluator reused; no reward, embedding, fuzzy matching, or post-result aliases",
        "input_hashes": {
            "task8_bundle": sha256(args.task8_bundle),
            "task6_bundle": sha256(args.task6_bundle),
            "targets": sha256(args.targets),
            "rule_map": sha256(args.rule_map),
            "normalization": sha256(args.normalization),
            "task7_evaluator": sha256(args.task7_evaluator),
            "protocol": sha256(args.protocol),
        },
        "input_audits": {
            "task8": audit["status"],
            "task6": task6_audit["status"],
            "task8_protocol_frozen": input_manifest[
                "protocol_frozen_before_model_outputs"
            ],
        },
        "bootstrap": {"samples": 10000, "seed": 20260726},
        "multiple_comparisons": (
            "Holm across four new levels, separately by population, "
            "perturbation family, and binary joint-exact outcome"
        ),
        "populations": populations,
    }
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "summary.md").write_text(markdown(summary), encoding="utf-8")
    manifest = {
        "status": "PASS",
        "files": [
            {
                "relative_path": name,
                "sha256": sha256(output_root / name),
                "size_bytes": (output_root / name).stat().st_size,
            }
            for name in ("summary.json", "summary.md")
        ],
    }
    (output_root / "evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
