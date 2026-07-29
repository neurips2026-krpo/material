#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import tarfile
import unicodedata
from typing import Any, Iterable


BUNDLE_SHA256 = "355c909adf3b830402a3b9f703c7a8e9d491c2ea9a09a1961cef76fa872c5eb5"
TARGET_SHA256 = "63c52792e1b2ea3f8865d25c87f6571c18924f62a033b607833bc234d92c5cc6"
INPUT_MANIFEST_SHA256 = (
    "31abc0be65062b4c26381569b3504da67d7f292c16e1c9675e51238d190f46d4"
)
BOOTSTRAP_SEED = 20260726
BOOTSTRAP_SAMPLES = 10_000

CONDITIONS = (
    "q14_base_nokg",
    "q14_base_clean",
    "q14_krpo_clean",
    "q14_base_cf",
    "q14_krpo_cf",
)
CF_CONDITIONS = {"q14_base_cf", "q14_krpo_cf"}
FAMILY_FIELDS = {
    "eight_principles": ("虚实", "寒热", "阴阳", "表里"),
    "etiology": ("病因", "子类"),
    "six_meridian": ("六经辨证证型",),
    "qi_blood_body_fluids": ("气血津液辨证证型",),
    "triple_burner": ("三焦辨证证型",),
    "wei_qi_ying_xue": ("卫气营血证型",),
    "zang_fu": ("脏腑辨证证型",),
}
FAMILY_ORDER = tuple(FAMILY_FIELDS)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_jsonl_bytes(payload: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        payload.decode("utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{line_number}: expected JSON object")
        rows.append(value)
    return rows


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return read_jsonl_bytes(path.read_bytes(), str(path))


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


class LabelNormalizer:
    def __init__(self, specification: dict[str, Any]) -> None:
        self.specification = specification
        self.outer = "".join(specification["strip_outer_characters"])
        self.replacements = specification["character_replacements"]
        escaped = "".join(
            re.escape(value) for value in specification["multi_label_separators"]
        )
        self.separator = re.compile(f"[{escaped}]")

    def component(self, value: str) -> str:
        result = unicodedata.normalize(
            self.specification["unicode_normalization"], value
        )
        for source, target in self.replacements.items():
            result = result.replace(source, target)
        if self.specification["remove_all_whitespace"]:
            result = re.sub(r"\s+", "", result)
        result = result.strip(self.outer)
        return result

    def normalize(self, value: Any) -> tuple[str, ...] | None:
        if not isinstance(value, str):
            return None
        normalized = self.component(value)
        if not normalized:
            return None
        parts = [
            self.component(part)
            for part in self.separator.split(normalized)
            if self.component(part)
        ]
        if not parts:
            return None
        if self.specification["deduplicate_multi_label_components"]:
            parts = list(dict.fromkeys(parts))
        if self.specification["sort_multi_label_components"]:
            parts = sorted(parts)
        return tuple(parts)


def json_objects(text: str) -> list[dict[str, Any]]:
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


def parse_final_summary(
    text: str, fields: tuple[str, ...]
) -> tuple[dict[str, str] | None, bool]:
    objects = json_objects(text)
    if not objects:
        return None, False
    final = objects[-1]
    expected_keys = set(fields) | {"综合结论"}
    valid = (
        set(final) == expected_keys
        and all(isinstance(final[key], str) for key in expected_keys)
    )
    if not valid:
        return None, False
    return {key: str(final[key]) for key in final}, True


ACTION_PATTERN = re.compile(r"(?i)action\s*(\d+)")


def parse_actions(text: str) -> tuple[dict[int, dict[str, Any]], list[int]]:
    markers = list(ACTION_PATTERN.finditer(text))
    actions: dict[int, dict[str, Any]] = {}
    duplicates: list[int] = []
    decoder = json.JSONDecoder()
    for offset, marker in enumerate(markers):
        action_id = int(marker.group(1))
        end = markers[offset + 1].start() if offset + 1 < len(markers) else len(text)
        segment = text[marker.end() : end]
        parsed: dict[str, Any] | None = None
        for position, character in enumerate(segment):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(segment[position:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                break
        if parsed is None:
            continue
        if action_id in actions:
            duplicates.append(action_id)
        else:
            actions[action_id] = parsed
    return actions, duplicates


def required_action_fields(reference_text: str) -> dict[int, tuple[str, ...]]:
    actions, duplicates = parse_actions(reference_text)
    if duplicates:
        raise ValueError(f"reference has duplicate action IDs: {duplicates}")
    required: dict[int, tuple[str, ...]] = {}
    for action_id, action in sorted(actions.items()):
        if action_id <= 2:
            continue
        if "辨证结果" in action:
            fields = ["辨证结果"]
        elif "分析结果" in action:
            fields = ["分析结果"]
        else:
            continue
        if "子类" in action:
            fields.append("子类")
        if not all(isinstance(action.get(field), str) for field in fields):
            raise ValueError(
                f"reference action {action_id} has non-string categorical field"
            )
        required[action_id] = tuple(fields)
    if not required:
        raise ValueError("reference trajectory has no categorical actions")
    return required


def normalized_summary(
    summary: dict[str, str] | None,
    fields: tuple[str, ...],
    normalizer: LabelNormalizer,
) -> tuple[tuple[str, tuple[str, ...] | None], ...] | None:
    if summary is None:
        return None
    return tuple((field, normalizer.normalize(summary.get(field))) for field in fields)


def evaluate_final(
    response: str,
    target: dict[str, str],
    fields: tuple[str, ...],
    normalizer: LabelNormalizer,
) -> dict[str, Any]:
    parsed, schema_valid = parse_final_summary(response, fields)
    field_correct: dict[str, bool] = {}
    target_normalized: dict[str, tuple[str, ...] | None] = {}
    predicted_normalized: dict[str, tuple[str, ...] | None] = {}
    for field in fields:
        expected = normalizer.normalize(target.get(field))
        predicted = normalizer.normalize(parsed.get(field)) if parsed else None
        target_normalized[field] = expected
        predicted_normalized[field] = predicted
        field_correct[field] = bool(
            schema_valid
            and expected is not None
            and predicted is not None
            and expected == predicted
        )
    return {
        "schema_valid": schema_valid,
        "joint_exact": schema_valid and all(field_correct.values()),
        "field_correct": field_correct,
        "target_normalized": {
            field: list(value) if value is not None else None
            for field, value in target_normalized.items()
        },
        "predicted_normalized": {
            field: list(value) if value is not None else None
            for field, value in predicted_normalized.items()
        },
        "parsed_summary": parsed,
    }


def evaluate_actions(
    response: str,
    reference_text: str,
    normalizer: LabelNormalizer,
) -> dict[str, Any]:
    reference_actions, reference_duplicates = parse_actions(reference_text)
    if reference_duplicates:
        raise ValueError(
            f"reference trajectory has duplicate actions: {reference_duplicates}"
        )
    required = required_action_fields(reference_text)
    predicted_actions, predicted_duplicates = parse_actions(response)
    steps: dict[str, Any] = {}
    complete_count = 0
    exact_step_count = 0
    field_total = 0
    field_exact = 0
    for action_id, fields in required.items():
        predicted = predicted_actions.get(action_id)
        complete = bool(
            predicted is not None
            and action_id not in predicted_duplicates
            and all(isinstance(predicted.get(field), str) for field in fields)
        )
        if complete:
            complete_count += 1
        correct: dict[str, bool] = {}
        for field in fields:
            expected_value = normalizer.normalize(
                reference_actions[action_id].get(field)
            )
            predicted_value = (
                normalizer.normalize(predicted.get(field))
                if complete and predicted is not None
                else None
            )
            is_correct = bool(
                complete
                and expected_value is not None
                and predicted_value is not None
                and expected_value == predicted_value
            )
            correct[field] = is_correct
            field_total += 1
            field_exact += int(is_correct)
        step_exact = complete and all(correct.values())
        exact_step_count += int(step_exact)
        steps[str(action_id)] = {
            "required_fields": list(fields),
            "complete": complete,
            "field_correct": correct,
            "step_exact": step_exact,
        }
    step_total = len(required)
    return {
        "required_action_ids": list(required),
        "duplicate_predicted_action_ids": predicted_duplicates,
        "complete_steps": complete_count,
        "required_steps": step_total,
        "action_completeness": complete_count / step_total,
        "exact_steps": exact_step_count,
        "action_step_exact": exact_step_count / step_total,
        "exact_fields": field_exact,
        "required_fields": field_total,
        "action_field_exact": field_exact / field_total,
        "action_trajectory_exact": (
            complete_count == step_total and exact_step_count == step_total
        ),
        "steps": steps,
    }


def accuracy(values: list[bool]) -> float:
    return sum(values) / len(values) if values else float("nan")


def wilson_interval(successes: int, total: int) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return [center - half, center + half]


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap_difference(
    candidate: list[bool],
    comparator: list[bool],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    if len(candidate) != len(comparator) or not candidate:
        raise ValueError("paired bootstrap requires equally sized non-empty arrays")
    rng = random.Random(seed)
    n = len(candidate)
    differences: list[float] = []
    for _ in range(samples):
        candidate_sum = 0
        comparator_sum = 0
        for _ in range(n):
            index = rng.randrange(n)
            candidate_sum += int(candidate[index])
            comparator_sum += int(comparator[index])
        differences.append((candidate_sum - comparator_sum) / n)
    differences.sort()
    return [
        percentile(differences, 0.025),
        percentile(differences, 0.975),
    ]


def exact_mcnemar(candidate: list[bool], comparator: list[bool]) -> dict[str, Any]:
    if len(candidate) != len(comparator):
        raise ValueError("McNemar inputs are not paired")
    candidate_only = sum(c and not r for c, r in zip(candidate, comparator))
    comparator_only = sum(r and not c for c, r in zip(candidate, comparator))
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


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    ordering = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(ordering):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def summarize_binary(values: list[bool]) -> dict[str, Any]:
    successes = sum(values)
    return {
        "correct": successes,
        "n": len(values),
        "rate": successes / len(values) if values else None,
        "wilson_95_ci": wilson_interval(successes, len(values)),
    }


def summarize_condition(
    rows: list[dict[str, Any]], metric_key: str
) -> dict[str, Any]:
    joints = [bool(row[metric_key]["joint_exact"]) for row in rows]
    schema = [bool(row[metric_key]["schema_valid"]) for row in rows]
    field_correct = [
        bool(value)
        for row in rows
        for value in row[metric_key]["field_correct"].values()
    ]
    actions = [row[f"{metric_key}_actions"] for row in rows]
    family_results: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        family_rows = [
            row for row in rows if row["metadata"]["task_family"] == family
        ]
        family_results[family] = summarize_binary(
            [bool(row[metric_key]["joint_exact"]) for row in family_rows]
        )
    family_macro = statistics.mean(
        result["rate"]
        for result in family_results.values()
        if result["rate"] is not None
    )
    action_steps = sum(int(row["exact_steps"]) for row in actions)
    required_steps = sum(int(row["required_steps"]) for row in actions)
    action_fields = sum(int(row["exact_fields"]) for row in actions)
    required_fields = sum(int(row["required_fields"]) for row in actions)
    trajectory = [bool(row["action_trajectory_exact"]) for row in actions]
    complete_steps = sum(int(row["complete_steps"]) for row in actions)
    etiology_rows = [
        row for row in rows if row["metadata"]["task_family"] == "etiology"
    ]
    etiology = {
        "parent": summarize_binary(
            [
                bool(row[metric_key]["field_correct"].get("病因", False))
                for row in etiology_rows
            ]
        ),
        "child": summarize_binary(
            [
                bool(row[metric_key]["field_correct"].get("子类", False))
                for row in etiology_rows
            ]
        ),
        "joint": summarize_binary(
            [bool(row[metric_key]["joint_exact"]) for row in etiology_rows]
        ),
    }
    return {
        "joint_exact": summarize_binary(joints),
        "schema_valid": summarize_binary(schema),
        "field_micro_exact": summarize_binary(field_correct),
        "family_macro_joint_exact": family_macro,
        "by_family": family_results,
        "etiology": etiology,
        "actions": {
            "complete_steps": complete_steps,
            "required_steps": required_steps,
            "action_completeness": complete_steps / required_steps,
            "exact_steps": action_steps,
            "action_step_exact": action_steps / required_steps,
            "exact_fields": action_fields,
            "required_fields": required_fields,
            "action_field_exact": action_fields / required_fields,
            "trajectory_exact": summarize_binary(trajectory),
        },
    }


def load_bundle(
    bundle: pathlib.Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    if sha256(bundle) != BUNDLE_SHA256:
        raise ValueError("Task 6 prediction bundle SHA-256 mismatch")
    checksum_path = pathlib.Path(str(bundle) + ".sha256")
    if not checksum_path.is_file():
        raise FileNotFoundError(f"missing checksum file: {checksum_path}")
    if checksum_path.read_text(encoding="utf-8").split()[0] != BUNDLE_SHA256:
        raise ValueError("Task 6 checksum sidecar mismatch")
    predictions: dict[str, list[dict[str, Any]]] = {}
    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
        if any(
            not member.isfile()
            or pathlib.PurePosixPath(member.name).is_absolute()
            or ".." in pathlib.PurePosixPath(member.name).parts
            for member in members
        ):
            raise ValueError("unsafe member in Task 6 prediction bundle")

        def payload(name: str) -> bytes:
            member = archive.getmember(name)
            handle = archive.extractfile(member)
            if handle is None:
                raise FileNotFoundError(name)
            return handle.read()

        manifest_payload = payload("task6_input_manifest.json")
        if sha256_bytes(manifest_payload) != INPUT_MANIFEST_SHA256:
            raise ValueError("Task 6 input manifest mismatch inside bundle")
        input_manifest = json.loads(manifest_payload)
        audit = json.loads(
            payload("server_runs/audits/prediction_manifest.json")
        )
        if audit.get("status") != "PASS":
            raise ValueError(f"Task 6 audit is not PASS: {audit.get('status')}")
        for condition in CONDITIONS:
            path = f"server_runs/predictions/{condition}/predictions.jsonl"
            rows = read_jsonl_bytes(payload(path), path)
            if len(rows) != 100:
                raise ValueError(f"{condition}: expected 100 predictions")
            predictions[condition] = rows
    return predictions, input_manifest, audit


def planned_comparisons(
    case_rows: list[dict[str, Any]],
    population_name: str,
) -> list[dict[str, Any]]:
    by_condition = {
        condition: {
            row["sample_id"]: row
            for row in case_rows
            if row["condition"] == condition
        }
        for condition in CONDITIONS
    }
    ids = sorted(by_condition["q14_base_nokg"])
    if not all(sorted(by_condition[condition]) == ids for condition in CONDITIONS):
        raise ValueError(f"{population_name}: condition IDs are not paired")
    specifications = [
        (
            "base_clean_minus_base_nokg_truth",
            "q14_base_clean",
            "q14_base_nokg",
            "truth",
        ),
        (
            "krpo_clean_minus_base_clean_truth",
            "q14_krpo_clean",
            "q14_base_clean",
            "truth",
        ),
        (
            "krpo_cf_minus_base_cf_adherence",
            "q14_krpo_cf",
            "q14_base_cf",
            "adherence",
        ),
        (
            "krpo_cf_minus_base_cf_truth",
            "q14_krpo_cf",
            "q14_base_cf",
            "truth",
        ),
    ]
    results: list[dict[str, Any]] = []
    for name, candidate_name, comparator_name, metric in specifications:
        candidate = [
            bool(by_condition[candidate_name][sample_id][metric]["joint_exact"])
            for sample_id in ids
        ]
        comparator = [
            bool(by_condition[comparator_name][sample_id][metric]["joint_exact"])
            for sample_id in ids
        ]
        mcnemar = exact_mcnemar(candidate, comparator)
        result = {
            "name": name,
            "population": population_name,
            "candidate": candidate_name,
            "comparator": comparator_name,
            "metric": metric,
            "n": len(ids),
            "candidate_accuracy": accuracy(candidate),
            "comparator_accuracy": accuracy(comparator),
            "difference": accuracy(candidate) - accuracy(comparator),
            "bootstrap_95_ci": paired_bootstrap_difference(
                candidate, comparator
            ),
            **mcnemar,
        }
        results.append(result)
    adjusted = holm_adjust([float(result["p_value_raw"]) for result in results])
    for result, value in zip(results, adjusted):
        result["p_value_holm"] = value
    return results


def switch_summary(
    rows: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    if model == "base":
        clean_condition = "q14_base_clean"
        cf_condition = "q14_base_cf"
    elif model == "krpo":
        clean_condition = "q14_krpo_clean"
        cf_condition = "q14_krpo_cf"
    else:
        raise ValueError(model)
    clean = {
        row["sample_id"]: row
        for row in rows
        if row["condition"] == clean_condition
    }
    cf = {
        row["sample_id"]: row
        for row in rows
        if row["condition"] == cf_condition
    }
    eligible = [
        sample_id
        for sample_id in sorted(clean)
        if cf[sample_id]["switch_targets_differ"]
    ]
    fixed_adoption = [
        bool(cf[sample_id]["adherence"]["joint_exact"]) for sample_id in eligible
    ]
    original_retention = [
        bool(cf[sample_id]["original"]["joint_exact"]) for sample_id in eligible
    ]
    clean_to_fixed: list[bool] = []
    any_change: list[bool] = []
    valid_pairs: list[bool] = []
    for sample_id in eligible:
        clean_row = clean[sample_id]
        cf_row = cf[sample_id]
        clean_vector = clean_row["truth"]["predicted_normalized"]
        cf_vector = cf_row["adherence"]["predicted_normalized"]
        valid = bool(
            clean_row["truth"]["schema_valid"]
            and cf_row["adherence"]["schema_valid"]
        )
        valid_pairs.append(valid)
        different = valid and clean_vector != cf_vector
        any_change.append(different)
        clean_to_fixed.append(
            bool(
                clean_row["truth"]["joint_exact"]
                and cf_row["adherence"]["joint_exact"]
                and different
            )
        )
    return {
        "model": model,
        "clean_condition": clean_condition,
        "counterfactual_condition": cf_condition,
        "eligible_cases": len(eligible),
        "fixed_target_adoption": summarize_binary(fixed_adoption),
        "original_target_retention": summarize_binary(original_retention),
        "paired_clean_to_fixed_switch": summarize_binary(clean_to_fixed),
        "any_prediction_change": summarize_binary(any_change),
        "valid_prediction_pair": summarize_binary(valid_pairs),
    }


def format_percent(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.1f}%"


def markdown_report(summary: dict[str, Any]) -> str:
    primary_name = next(
        name
        for name in summary["populations"]
        if name.startswith("primary_")
    )
    primary = summary["populations"][primary_name]
    primary_cases = primary["cases"]
    lines = [
        "# Task 7 reward-independent TCM evaluation",
        "",
        f"## Primary {primary_cases}-case results",
        "",
        "| Condition | Truth joint | Schema valid | Action-step exact | Action complete |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        result = primary["conditions"][condition]["truth"]
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    format_percent(result["joint_exact"]["rate"]),
                    format_percent(result["schema_valid"]["rate"]),
                    format_percent(result["actions"]["action_step_exact"]),
                    format_percent(result["actions"]["action_completeness"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"## Counterfactual context adherence ({primary_cases} cases)",
            "",
            "| Condition | Fixed-target joint | Action-step exact | Truth joint under CF |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition in ("q14_base_cf", "q14_krpo_cf"):
        condition_result = primary["conditions"][condition]
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    format_percent(
                        condition_result["adherence"]["joint_exact"]["rate"]
                    ),
                    format_percent(
                        condition_result["adherence"]["actions"][
                            "action_step_exact"
                        ]
                    ),
                    format_percent(
                        condition_result["truth"]["joint_exact"]["rate"]
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"## Planned paired comparisons ({primary_cases} cases)",
            "",
            "| Comparison | Candidate | Comparator | Δ pp | 95% CI pp | McNemar p | Holm p |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in primary["comparisons"]:
        ci = result["bootstrap_95_ci"]
        lines.append(
            "| "
            + " | ".join(
                [
                    result["name"],
                    format_percent(result["candidate_accuracy"]),
                    format_percent(result["comparator_accuracy"]),
                    f"{100 * result['difference']:.1f}",
                    f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]",
                    f"{result['p_value_raw']:.4g}",
                    f"{result['p_value_holm']:.4g}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "All automatic labels use frozen normalized exact matching. "
            "No embedding, fuzzy matching, submitted reward, or free-text "
            "conclusion score is used.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--targets", type=pathlib.Path, required=True)
    parser.add_argument("--normalization", type=pathlib.Path, required=True)
    parser.add_argument("--protocol", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--expected-target-sha256",
        default=TARGET_SHA256,
        help="Expected target hash; override only for a documented source audit.",
    )
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    targets_path = args.targets.resolve()
    normalization_path = args.normalization.resolve()
    protocol_path = args.protocol.resolve()
    output_root = args.output_root.resolve()
    if sha256(targets_path) != args.expected_target_sha256:
        raise ValueError("Task 6 targets SHA-256 mismatch")
    normalization_spec = read_json(normalization_path)
    normalizer = LabelNormalizer(normalization_spec)
    predictions, input_manifest, audit = load_bundle(bundle)
    targets = read_jsonl(targets_path)
    if len(targets) != 100:
        raise ValueError("expected 100 target records")
    targets_by_id = {str(row["sample_id"]): row for row in targets}
    if len(targets_by_id) != 100:
        raise ValueError("duplicate target sample IDs")
    primary_count = sum(
        bool(row["metadata"]["primary_leakage_free"]) for row in targets
    )

    case_metrics: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for prediction in predictions[condition]:
            sample_id = str(prediction["sample_id"])
            target_row = targets_by_id[sample_id]
            metadata = target_row["metadata"]
            family = str(metadata["task_family"])
            fields = FAMILY_FIELDS[family]
            response = str(prediction["response"])
            truth_target = target_row["final_summaries"]["clean"]
            truth = evaluate_final(
                response, truth_target, fields, normalizer
            )
            truth_actions = evaluate_actions(
                response, target_row["clean_output"], normalizer
            )
            record: dict[str, Any] = {
                "sample_id": sample_id,
                "condition": condition,
                "response_sha256": prediction["response_sha256"],
                "metadata": metadata,
                "truth": truth,
                "truth_actions": truth_actions,
                "adherence": None,
                "adherence_actions": None,
                "original": None,
                "switch_targets_differ": None,
            }
            if condition in CF_CONDITIONS:
                fixed_target = target_row["final_summaries"][
                    "counterfactual_fixed"
                ]
                original_target = target_row["final_summaries"][
                    "counterfactual_original"
                ]
                record["adherence"] = evaluate_final(
                    response, fixed_target, fields, normalizer
                )
                record["adherence_actions"] = evaluate_actions(
                    response,
                    target_row["counterfactual_fixed_output"],
                    normalizer,
                )
                record["original"] = evaluate_final(
                    response, original_target, fields, normalizer
                )
                original_vector = normalized_summary(
                    original_target, fields, normalizer
                )
                fixed_vector = normalized_summary(
                    fixed_target, fields, normalizer
                )
                record["switch_targets_differ"] = original_vector != fixed_vector
            case_metrics.append(record)

    expected_records = len(CONDITIONS) * len(targets)
    if len(case_metrics) != expected_records:
        raise ValueError("Task 7 case metric count mismatch")
    write_jsonl(output_root / "case_metrics.jsonl", case_metrics)

    populations: dict[str, Any] = {}
    for population_name, primary_only in (
        (f"primary_{primary_count}", True),
        ("continuity_100", False),
    ):
        population_rows = [
            row
            for row in case_metrics
            if not primary_only or row["metadata"]["primary_leakage_free"]
        ]
        conditions: dict[str, Any] = {}
        for condition in CONDITIONS:
            condition_rows = [
                row for row in population_rows if row["condition"] == condition
            ]
            conditions[condition] = {
                "truth": summarize_condition(condition_rows, "truth")
            }
            if condition in CF_CONDITIONS:
                conditions[condition]["adherence"] = summarize_condition(
                    condition_rows, "adherence"
                )
        populations[population_name] = {
            "cases": primary_count if primary_only else 100,
            "prediction_records": len(population_rows),
            "conditions": conditions,
            "comparisons": planned_comparisons(
                population_rows, population_name
            ),
            "switch": {
                "base": switch_summary(population_rows, "base"),
                "krpo": switch_summary(population_rows, "krpo"),
            },
        }

    summary = {
        "status": "PASS",
        "verification_status": "ANALYZED",
        "evaluator": "reward-independent standard-library exact evaluator",
        "input_hashes": {
            "prediction_bundle": sha256(bundle),
            "targets": sha256(targets_path),
            "input_manifest": INPUT_MANIFEST_SHA256,
            "normalization_map": sha256(normalization_path),
            "protocol": sha256(protocol_path),
        },
        "input_audit_status": audit["status"],
        "normalization": normalization_spec,
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "interval": "paired case percentile 95% CI",
        },
        "multiple_comparisons": (
            "Holm correction across four predeclared comparisons, "
            "separately within each population"
        ),
        "populations": populations,
    }
    write_json(output_root / "summary.json", summary)
    (output_root / "summary.md").write_text(
        markdown_report(summary), encoding="utf-8"
    )

    csv_path = output_root / "case_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "condition",
                "task_family",
                "primary_leakage_free",
                "truth_joint_exact",
                "truth_schema_valid",
                "truth_action_completeness",
                "truth_action_step_exact",
                "adherence_joint_exact",
                "adherence_schema_valid",
                "adherence_action_completeness",
                "adherence_action_step_exact",
                "response_sha256",
            ],
        )
        writer.writeheader()
        for row in case_metrics:
            adherence = row["adherence"]
            adherence_actions = row["adherence_actions"]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "condition": row["condition"],
                    "task_family": row["metadata"]["task_family"],
                    "primary_leakage_free": row["metadata"][
                        "primary_leakage_free"
                    ],
                    "truth_joint_exact": row["truth"]["joint_exact"],
                    "truth_schema_valid": row["truth"]["schema_valid"],
                    "truth_action_completeness": row["truth_actions"][
                        "action_completeness"
                    ],
                    "truth_action_step_exact": row["truth_actions"][
                        "action_step_exact"
                    ],
                    "adherence_joint_exact": (
                        adherence["joint_exact"] if adherence else ""
                    ),
                    "adherence_schema_valid": (
                        adherence["schema_valid"] if adherence else ""
                    ),
                    "adherence_action_completeness": (
                        adherence_actions["action_completeness"]
                        if adherence_actions
                        else ""
                    ),
                    "adherence_action_step_exact": (
                        adherence_actions["action_step_exact"]
                        if adherence_actions
                        else ""
                    ),
                    "response_sha256": row["response_sha256"],
                }
            )

    output_files = [
        output_root / "case_metrics.jsonl",
        output_root / "case_audit.csv",
        output_root / "summary.json",
        output_root / "summary.md",
    ]
    manifest = {
        "status": "PASS",
        "files": [
            {
                "path": str(path),
                "records": (
                    sum(1 for _ in path.open(encoding="utf-8"))
                    if path.suffix in {".jsonl", ".csv"}
                    else None
                ),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in output_files
        ],
    }
    write_json(output_root / "evaluation_manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
