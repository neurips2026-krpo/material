#!/usr/bin/env python3
"""Independent verification of Task 8 joint-exact curves and statistics."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import pathlib
import random
import re
import tarfile
import unicodedata
from typing import Any


TASK6_BUNDLE_SHA256 = (
    "355c909adf3b830402a3b9f703c7a8e9d491c2ea9a09a1961cef76fa872c5eb5"
)
TARGET_SHA256 = (
    "63c52792e1b2ea3f8865d25c87f6571c18924f62a033b607833bc234d92c5cc6"
)
LEVELS = (100, 75, 50, 25, 0)
NEW_LEVELS = (75, 50, 25, 0)
MODES = ("relevant", "random", "conflict")
FIELDS = {
    "eight_principles": ("虚实", "寒热", "阴阳", "表里"),
    "etiology": ("病因", "子类"),
    "six_meridian": ("六经辨证证型",),
    "qi_blood_body_fluids": ("气血津液辨证证型",),
    "triple_burner": ("三焦辨证证型",),
    "wei_qi_ying_xue": ("卫气营血证型",),
    "zang_fu": ("脏腑辨证证型",),
}


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, str):
        return None
    value = unicodedata.normalize("NFKC", value).replace("淤", "瘀")
    value = re.sub(r"\s+", "", value).strip("\"'“”‘’[]【】")
    if not value:
        return None
    parts = [
        part.strip("\"'“”‘’[]【】")
        for part in re.split(r"[,，、;；/／]", value)
        if part.strip("\"'“”‘’[]【】")
    ]
    return tuple(sorted(set(parts))) if parts else None


def final_object(text: str, fields: tuple[str, ...]) -> dict[str, str] | None:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    if not objects:
        return None
    result = objects[-1]
    expected = set(fields) | {"综合结论"}
    if set(result) != expected or not all(
        isinstance(result.get(field), str) for field in expected
    ):
        return None
    return {field: str(result[field]) for field in result}


def exact(response: str, target: dict[str, str], fields: tuple[str, ...]) -> bool:
    prediction = final_object(response, fields)
    return bool(
        prediction is not None
        and all(
            normalize(prediction.get(field)) == normalize(target.get(field))
            and normalize(target.get(field)) is not None
            for field in fields
        )
    )


def exact_mcnemar(candidate: list[bool], comparator: list[bool]) -> float:
    a = sum(c and not r for c, r in zip(candidate, comparator))
    b = sum(r and not c for c, r in zip(candidate, comparator))
    total = a + b
    if total == 0:
        return 1.0
    tail = sum(
        math.comb(total, value)
        for value in range(min(a, b) + 1)
    ) / (2**total)
    return min(1.0, 2 * tail)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def bootstrap(candidate: list[bool], comparator: list[bool]) -> list[float]:
    rng = random.Random(20260726)
    size = len(candidate)
    values: list[float] = []
    for _ in range(10_000):
        indices = [rng.randrange(size) for _ in range(size)]
        values.append(
            (
                sum(candidate[index] for index in indices)
                - sum(comparator[index] for index in indices)
            )
            / size
        )
    return [quantile(values, 0.025), quantile(values, 0.975)]


def holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def load_archive_rows(
    bundle: pathlib.Path, paths: list[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    with tarfile.open(bundle, "r:gz") as archive:
        for condition in paths:
            name = f"server_runs/predictions/{condition}/predictions.jsonl"
            handle = archive.extractfile(name)
            if handle is None:
                raise FileNotFoundError(name)
            rows = [
                json.loads(line)
                for line in handle.read().decode("utf-8").splitlines()
                if line.strip()
            ]
            if len(rows) != 100:
                raise ValueError(f"{condition}: expected 100 rows")
            result[condition] = {str(row["sample_id"]): row for row in rows}
    return result


def condition(model: str, mode: str, level: int) -> str:
    return (
        f"q14_{model}_clean"
        if level == 100
        else f"q14_{model}_{mode}_q{level:03d}"
    )


def assert_close(observed: Any, expected: Any, label: str) -> None:
    if isinstance(observed, list) and isinstance(expected, list):
        if len(observed) != len(expected):
            raise ValueError(f"{label}: list length mismatch")
        for index, (left, right) in enumerate(zip(observed, expected)):
            assert_close(left, right, f"{label}[{index}]")
        return
    if isinstance(observed, (int, float)) and isinstance(expected, (int, float)):
        if not math.isclose(float(observed), float(expected), abs_tol=1e-15):
            raise ValueError(f"{label}: {observed} != {expected}")
        return
    if observed != expected:
        raise ValueError(f"{label}: {observed} != {expected}")


def compare(
    candidate: list[bool],
    comparator: list[bool],
    stored: dict[str, Any],
    label: str,
) -> None:
    difference = sum(candidate) / len(candidate) - sum(comparator) / len(comparator)
    assert_close(stored["difference"], difference, f"{label}:difference")
    assert_close(
        stored["p_value_raw"],
        exact_mcnemar(candidate, comparator),
        f"{label}:p",
    )
    assert_close(
        stored["bootstrap_95_ci"],
        bootstrap(candidate, comparator),
        f"{label}:ci",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task8-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.task8_root.resolve()
    task6 = root.parent / "task6"
    task7 = root.parent / "task7"
    bundle8 = root / "task8_prediction_bundle.tar.gz"
    bundle6 = task6 / "task6_prediction_bundle.tar.gz"
    targets_path = task6 / "targets/tcm_targets_100.jsonl"
    if digest(bundle6) != TASK6_BUNDLE_SHA256:
        raise ValueError("Task 6 bundle identity mismatch")
    if digest(targets_path) != TARGET_SHA256:
        raise ValueError("target identity mismatch")
    expected8 = pathlib.Path(str(bundle8) + ".sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    if digest(bundle8) != expected8:
        raise ValueError("Task 8 bundle identity mismatch")

    evaluator_tree = ast.parse(
        (root / "evaluate_task8.py").read_text(encoding="utf-8")
    )
    imports: set[str] = set()
    for node in ast.walk(evaluator_tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    allowed = {
        "__future__", "argparse", "hashlib", "importlib", "json", "pathlib",
        "tarfile", "typing",
    }
    if imports - allowed:
        raise ValueError(f"unexpected evaluator imports: {imports - allowed}")
    source = (root / "evaluate_task8.py").read_text(encoding="utf-8").lower()
    if any(
        term in source
        for term in ("tcm_reward", "embedding api", "fuzzywuzzy", "levenshtein")
    ):
        raise ValueError("forbidden reward or semantic dependency")

    new_conditions = [
        f"q14_{model}_{mode}_q{level:03d}"
        for mode in MODES
        for level in NEW_LEVELS
        for model in ("base", "krpo")
    ]
    anchor_conditions = [
        "q14_base_clean", "q14_krpo_clean", "q14_base_nokg",
        "q14_base_cf", "q14_krpo_cf",
    ]
    predictions = load_archive_rows(bundle8, new_conditions)
    predictions.update(load_archive_rows(bundle6, anchor_conditions))
    targets = {str(row["sample_id"]): row for row in records(targets_path)}
    maps = {
        str(row["sample_id"]): row
        for row in records(root / "relevant_rule_map_100.jsonl")
    }
    summary = json.loads(
        (root / "results/summary.json").read_text(encoding="utf-8")
    )

    outcomes: dict[tuple[str, str, str], bool] = {}
    for condition_name, by_id in predictions.items():
        for sample_id, prediction in by_id.items():
            target = targets[sample_id]
            fields = FIELDS[target["metadata"]["task_family"]]
            outcomes[(condition_name, sample_id, "truth")] = exact(
                str(prediction["response"]),
                target["final_summaries"]["clean"],
                fields,
            )
            if "_conflict_" in condition_name or condition_name.endswith("_cf"):
                outcomes[(condition_name, sample_id, "adherence")] = exact(
                    str(prediction["response"]),
                    target["final_summaries"]["counterfactual_fixed"],
                    fields,
                )

    comparisons_checked = 0
    rates_checked = 0
    for population_name, primary_only in (
        ("primary", True),
        ("continuity", False),
    ):
        population = summary["populations"][population_name]
        for mode in MODES:
            eligible_key = (
                "eligible_conflict"
                if mode == "conflict"
                else "eligible_relevant_deletion"
            )
            ids = [
                sample_id
                for sample_id in sorted(targets)
                if maps[sample_id][eligible_key]
                and (
                    not primary_only
                    or targets[sample_id]["metadata"]["primary_leakage_free"]
                )
            ]
            curve = population["curves"][mode]
            if curve["eligible_cases"] != len(ids):
                raise ValueError(f"{population_name}/{mode}: eligible n mismatch")
            for model in ("base", "krpo"):
                clean = [
                    outcomes[(condition(model, mode, 100), sample_id, "truth")]
                    for sample_id in ids
                ]
                for level in LEVELS:
                    values = [
                        outcomes[(condition(model, mode, level), sample_id, "truth")]
                        for sample_id in ids
                    ]
                    stored = curve["models"][model][str(level)]["truth"][
                        "joint_exact"
                    ]
                    assert_close(stored["correct"], sum(values), "correct")
                    assert_close(stored["n"], len(values), "n")
                    assert_close(stored["rate"], sum(values) / len(values), "rate")
                    rates_checked += 1
                    if mode == "conflict" and level != 100:
                        adherence = [
                            outcomes[
                                (
                                    condition(model, mode, level),
                                    sample_id,
                                    "adherence",
                                )
                            ]
                            for sample_id in ids
                        ]
                        stored_adherence = curve["models"][model][str(level)][
                            "adherence"
                        ]["joint_exact"]
                        assert_close(
                            stored_adherence["rate"],
                            sum(adherence) / len(adherence),
                            "adherence rate",
                        )
                        rates_checked += 1
                for item in curve["degradation_from_clean_truth"][model]:
                    level = item["level"]
                    current = [
                        outcomes[
                            (condition(model, mode, level), sample_id, "truth")
                        ]
                        for sample_id in ids
                    ]
                    compare(current, clean, item, item["name"])
                    comparisons_checked += 1
            raw_truth: list[float] = []
            for item in curve["model_comparisons_truth"]:
                level = item["level"]
                candidate = [
                    outcomes[
                        (condition("krpo", mode, level), sample_id, "truth")
                    ]
                    for sample_id in ids
                ]
                comparator = [
                    outcomes[
                        (condition("base", mode, level), sample_id, "truth")
                    ]
                    for sample_id in ids
                ]
                compare(candidate, comparator, item, item["name"])
                raw_truth.append(item["p_value_raw"])
                comparisons_checked += 1
            assert_close(
                [item["p_value_holm"] for item in curve["model_comparisons_truth"]],
                holm(raw_truth),
                f"{population_name}/{mode}:truth Holm",
            )
            if mode == "conflict":
                raw_adherence: list[float] = []
                for item in curve["model_comparisons_adherence"]:
                    level = item["level"]
                    candidate = [
                        outcomes[
                            (
                                condition("krpo", mode, level),
                                sample_id,
                                "adherence",
                            )
                        ]
                        for sample_id in ids
                    ]
                    comparator = [
                        outcomes[
                            (
                                condition("base", mode, level),
                                sample_id,
                                "adherence",
                            )
                        ]
                        for sample_id in ids
                    ]
                    compare(candidate, comparator, item, item["name"])
                    raw_adherence.append(item["p_value_raw"])
                    comparisons_checked += 1
                assert_close(
                    [
                        item["p_value_holm"]
                        for item in curve["model_comparisons_adherence"]
                    ],
                    holm(raw_adherence),
                    f"{population_name}/{mode}:adherence Holm",
                )

        deletion_ids = [
            sample_id
            for sample_id in sorted(targets)
            if maps[sample_id]["eligible_relevant_deletion"]
            and (
                not primary_only
                or targets[sample_id]["metadata"]["primary_leakage_free"]
            )
        ]
        controls = population[
            "amount_matched_relevant_minus_random_truth"
        ]
        for model in ("base", "krpo"):
            raw: list[float] = []
            for item in controls[model]:
                level = item["level"]
                candidate = [
                    outcomes[
                        (
                            condition(model, "relevant", level),
                            sample_id,
                            "truth",
                        )
                    ]
                    for sample_id in deletion_ids
                ]
                comparator = [
                    outcomes[
                        (
                            condition(model, "random", level),
                            sample_id,
                            "truth",
                        )
                    ]
                    for sample_id in deletion_ids
                ]
                compare(candidate, comparator, item, item["name"])
                raw.append(item["p_value_raw"])
                comparisons_checked += 1
            assert_close(
                [item["p_value_holm"] for item in controls[model]],
                holm(raw),
                f"{population_name}/{model}:control Holm",
            )

    report = {
        "status": "PASS",
        "verification_status": "VERIFIED",
        "independent_parser": True,
        "independently_checked_joint_rates": rates_checked,
        "independently_checked_paired_comparisons": comparisons_checked,
        "bootstrap_replicates_per_comparison": 10000,
        "evaluator_standard_library_only": True,
        "reward_dependency": False,
        "embedding_dependency": False,
        "task8_bundle_sha256": digest(bundle8),
        "task6_bundle_sha256": digest(bundle6),
        "targets_sha256": digest(targets_path),
        "summary_sha256": digest(root / "results/summary.json"),
    }
    output = root / "results/independent_verification.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
