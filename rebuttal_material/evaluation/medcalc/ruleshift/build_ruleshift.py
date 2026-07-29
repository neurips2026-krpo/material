#!/usr/bin/env python3
"""Build paired control/rule-revision MedCalc inputs without model outputs.

For four fully recomputable seen calculators, one active criterion is selected
deterministically and its weight is increased by two points. Cases with no
active criterion are excluded before inference.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import pathlib
import re
from typing import Any


SUPPORTED = {
    4: "CHA2DS2-VASc",
    8: "Wells PE",
    18: "HEART",
    45: "CURB-65",
}
DELTA = 2.0
BUILD_SEED = 20260727


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        rows.append(value)
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects = []
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


def parse_actions(text: str) -> dict[int, dict[str, Any]]:
    markers = list(re.finditer(r"(?im)^action(\d+)\s*:", text))
    actions: dict[int, dict[str, Any]] = {}
    for offset, marker in enumerate(markers):
        end = markers[offset + 1].start() if offset + 1 < len(markers) else len(text)
        objects = json_objects(text[marker.end() : end])
        if objects:
            actions[int(marker.group(1))] = objects[0]
    return actions


def number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric value")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, (list, tuple)) and value:
        result = float(value[0])
    else:
        result = float(str(value))
    if not math.isfinite(result):
        raise ValueError(f"non-finite number: {value}")
    return result


def boolean(variables: dict[str, Any], key: str) -> bool:
    value = variables.get(key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in {"true", "yes", "present"}:
            return True
        if value.strip().lower() in {"false", "no", "absent"}:
            return False
    return bool(value)


def add(
    values: list[dict[str, Any]],
    criterion_id: str,
    label: str,
    weight: float,
) -> None:
    if weight > 0:
        values.append(
            {
                "criterion_id": criterion_id,
                "criterion_label": label,
                "old_weight": float(weight),
            }
        )


def cha2ds2_vasc(variables: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    age = number(variables["age"])
    add(
        result,
        "age",
        "Age 75 years or older",
        2 if age >= 75 else 1 if age >= 65 else 0,
    )
    add(
        result,
        "female_sex",
        "Female sex",
        1 if str(variables["sex"]).strip().lower() == "female" else 0,
    )
    add(
        result,
        "chf",
        "Congestive heart failure history",
        1 if boolean(variables, "Congestive Heart Failure") else 0,
    )
    add(
        result,
        "hypertension",
        "Hypertension history",
        1 if boolean(variables, "Hypertension history") else 0,
    )
    vascular_event = any(
        boolean(variables, key)
        for key in (
            "Stroke",
            "Transient Ischemic Attacks History",
            "Thromboembolism history",
        )
    )
    add(
        result,
        "stroke_tia_thromboembolism",
        "Stroke, TIA, or thromboembolism history",
        2 if vascular_event else 0,
    )
    add(
        result,
        "vascular_disease",
        "Vascular disease history",
        1 if boolean(variables, "Vascular disease history") else 0,
    )
    add(
        result,
        "diabetes",
        "Diabetes history",
        1 if boolean(variables, "Diabetes history") else 0,
    )
    return result


def wells_pe(variables: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    add(
        result,
        "dvt_signs",
        "Clinical signs and symptoms of DVT",
        3
        if boolean(
            variables, "Clinical signs and symptoms of Deep Vein Thrombosis"
        )
        else 0,
    )
    add(
        result,
        "pe_most_likely",
        "PE is the number-one diagnosis or equally likely",
        3
        if boolean(
            variables,
            "Pulmonary Embolism is #1 diagnosis OR equally likely",
        )
        else 0,
    )
    add(
        result,
        "heart_rate",
        "Heart rate above 100 beats per minute",
        1.5 if number(variables["Heart Rate or Pulse"]) > 100 else 0,
    )
    add(
        result,
        "immobilization_or_surgery",
        "Immobilization at least 3 days or surgery in the previous 4 weeks",
        1.5
        if boolean(variables, "Immobilization for at least 3 days")
        or boolean(variables, "Surgery in the previous 4 weeks")
        else 0,
    )
    add(
        result,
        "previous_pe_or_dvt",
        "Previous objectively diagnosed PE or DVT",
        1.5
        if boolean(variables, "Previously Documented Pulmonary Embolism")
        or boolean(variables, "Previously documented Deep Vein Thrombosis")
        else 0,
    )
    add(
        result,
        "hemoptysis",
        "Hemoptysis",
        1 if boolean(variables, "Hemoptysis") else 0,
    )
    add(
        result,
        "malignancy",
        "Malignancy with recent treatment or palliative care",
        1
        if boolean(
            variables,
            "Malignancy with treatment within 6 months or palliative",
        )
        else 0,
    )
    return result


def heart(variables: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    history = str(variables["Suspicion History"]).strip().lower()
    history_score = {
        "slightly suspicious": 0,
        "moderately suspicious": 1,
        "highly suspicious": 2,
    }[history]
    add(result, "history", f"History: {variables['Suspicion History']}", history_score)

    ecg = str(variables.get("Electrocardiogram Test", "Normal")).strip().lower()
    ecg_score = {
        "normal": 0,
        "non-specific repolarization disturbance": 1,
        "significant st deviation": 2,
    }[ecg]
    add(
        result,
        "ecg",
        f"ECG: {variables.get('Electrocardiogram Test', 'Normal')}",
        ecg_score,
    )

    age = number(variables["age"])
    add(
        result,
        "age",
        "Age 65 years or older" if age >= 65 else "Age 45 to 64 years",
        2 if age >= 65 else 1 if age >= 45 else 0,
    )

    atherosclerotic = boolean(variables, "atherosclerotic disease") or boolean(
        variables, "Transient Ischemic Attacks History"
    )
    risk_count = sum(
        boolean(variables, key)
        for key in (
            "Hypertension history",
            "hypercholesterolemia",
            "Diabetes mellitus",
            "obesity",
            "smoking",
            "parent or sibling with Cardiovascular disease before age 65",
        )
    )
    risk_score = 2 if atherosclerotic or risk_count >= 3 else 1 if risk_count else 0
    add(
        result,
        "risk_factors",
        "HEART cardiovascular risk-factor criterion",
        risk_score,
    )

    troponin = str(
        variables.get("Initial troponin", "less than or equal to normal limit")
    ).strip().lower()
    troponin_score = {
        "less than or equal to normal limit": 0,
        "between the normal limit or up to three times the normal limit": 1,
        "greater than three times normal limit": 2,
    }[troponin]
    add(
        result,
        "troponin",
        f"Initial troponin: {variables.get('Initial troponin', 'normal')}",
        troponin_score,
    )
    return result


def curb65(variables: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    add(
        result,
        "confusion",
        "Confusion",
        1 if boolean(variables, "Confusion") else 0,
    )
    bun = variables["Blood Urea Nitrogen (BUN)"]
    bun_value = number(bun)
    bun_unit = str(bun[1]).strip().lower() if isinstance(bun, list) else "mg/dl"
    bun_active = bun_value > (7 if "mmol" in bun_unit else 19)
    add(
        result,
        "bun",
        "BUN above 19 mg/dL or urea above 7 mmol/L",
        1 if bun_active else 0,
    )
    add(
        result,
        "respiratory_rate",
        "Respiratory rate at least 30 per minute",
        1 if number(variables["respiratory rate"]) >= 30 else 0,
    )
    blood_pressure_active = (
        number(variables["Systolic Blood Pressure"]) < 90
        or number(variables["Diastolic Blood Pressure"]) <= 60
    )
    add(
        result,
        "blood_pressure",
        "Systolic BP below 90 or diastolic BP at most 60 mmHg",
        1 if blood_pressure_active else 0,
    )
    add(
        result,
        "age",
        "Age 65 years or older",
        1 if number(variables["age"]) >= 65 else 0,
    )
    return result


SCORERS = {
    4: cha2ds2_vasc,
    8: wells_pe,
    18: heart,
    45: curb65,
}


def display_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def insert_rule_block(instruction: str, block: str) -> str:
    marker = "</rule_card>"
    if instruction.count(marker) != 1:
        raise ValueError("instruction must contain exactly one </rule_card>")
    return instruction.replace(marker, f"\n{block}\n\n{marker}", 1)


def select_active(
    sample_id: str, active: list[dict[str, Any]]
) -> dict[str, Any]:
    digest = hashlib.sha256(
        f"{BUILD_SEED}:{sample_id}".encode("utf-8")
    ).digest()
    index = int.from_bytes(digest[:8], "big") % len(active)
    return dict(active[index])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    root = args.output_root.resolve()
    rows = read_jsonl(source)
    source_by_calculator = collections.Counter(
        int(row["metadata"]["calculator_id"]) for row in rows
    )
    if any(source_by_calculator[calculator] != 20 for calculator in SUPPORTED):
        raise ValueError("expected exactly 20 held-out cases per supported calculator")

    inputs: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    criteria_counts: collections.Counter[str] = collections.Counter()
    retained_counts: collections.Counter[int] = collections.Counter()

    for source_row in rows:
        calculator_id = int(source_row["metadata"]["calculator_id"])
        if calculator_id not in SUPPORTED:
            continue
        actions = parse_actions(str(source_row["output"]))
        if 1 not in actions:
            raise ValueError(f"{source_row['sample_id']}: missing reference action1")
        variables = actions[1].get("extracted_variables")
        if not isinstance(variables, dict):
            raise ValueError(f"{source_row['sample_id']}: invalid extracted variables")
        active = SCORERS[calculator_id](variables)
        original_answer = number(source_row["evaluation_target"]["answer"])
        recomputed = sum(float(item["old_weight"]) for item in active)
        if not math.isclose(recomputed, original_answer, abs_tol=1e-9):
            raise ValueError(
                f"{source_row['sample_id']}: recomputed {recomputed} "
                f"does not match official {original_answer}"
            )
        if not active:
            exclusions.append(
                {
                    "sample_id": source_row["sample_id"],
                    "calculator_id": calculator_id,
                    "reason": "official score is zero; no active criterion to revise",
                }
            )
            continue

        selected = select_active(str(source_row["sample_id"]), active)
        selected["new_weight"] = float(selected["old_weight"]) + DELTA
        selected["delta"] = DELTA
        revised_answer = original_answer + DELTA
        revision_id = (
            f"rs-{calculator_id}-{selected['criterion_id']}-"
            f"{sha256_bytes(str(source_row['sample_id']).encode())[:8]}"
        )
        control_id = f"{source_row['sample_id']}--control"
        revision_sample_id = f"{source_row['sample_id']}--revision"
        control_block = f"""<current_rule_revision>
Controlled rule-version notice ({revision_id}).
For this matched control prompt, the numerical weight of criterion
\"{selected['criterion_label']}\" remains {display_number(selected['old_weight'])}
points. All other supplied criteria and the requested structured output remain
unchanged. Apply the supplied rule rather than a memorized alternative.
</current_rule_revision>"""
        revision_block = f"""<current_rule_revision>
Synthetic controlled rule update ({revision_id}); this is not a clinical recommendation.
For this task only, criterion \"{selected['criterion_label']}\" now awards
{display_number(selected['new_weight'])} points instead of
{display_number(selected['old_weight'])} points. This single updated weight
supersedes the conflicting weight in the official implementation above. All
other criteria are unchanged. Recompute action3, action4, and the final summary
under this current rule; do not return the legacy official score.
</current_rule_revision>"""
        shared_metadata = {
            "base_sample_id": source_row["sample_id"],
            "calculator_id": calculator_id,
            "calculator_name": source_row["metadata"]["calculator_name"],
            "evaluation_split": source_row["metadata"]["evaluation_split"],
            "revision_id": revision_id,
            "criterion_id": selected["criterion_id"],
            "criterion_label": selected["criterion_label"],
            "old_weight": selected["old_weight"],
            "new_weight": selected["new_weight"],
            "delta": DELTA,
            "contains_gold_answer": False,
        }
        inputs.extend(
            [
                {
                    "sample_id": control_id,
                    "instruction": insert_rule_block(
                        str(source_row["instruction"]), control_block
                    ),
                    "metadata": {**shared_metadata, "variant": "control"},
                },
                {
                    "sample_id": revision_sample_id,
                    "instruction": insert_rule_block(
                        str(source_row["instruction"]), revision_block
                    ),
                    "metadata": {**shared_metadata, "variant": "revision"},
                },
            ]
        )
        targets.append(
            {
                "base_sample_id": source_row["sample_id"],
                "control_sample_id": control_id,
                "revision_sample_id": revision_sample_id,
                "metadata": {
                    **shared_metadata,
                    "source_note_id": source_row["metadata"]["note_id"],
                },
                "original_answer": display_number(original_answer),
                "revised_answer": display_number(revised_answer),
                "expected_extracted_variables": variables,
            }
        )
        retained_counts[calculator_id] += 1
        criteria_counts[
            f"{calculator_id}:{selected['criterion_id']}"
        ] += 1

    if len(targets) != 67 or len(inputs) != 134 or len(exclusions) != 13:
        raise ValueError(
            f"unexpected retained/input/excluded counts: "
            f"{len(targets)}/{len(inputs)}/{len(exclusions)}"
        )
    input_path = root / "inputs/medcalc_ruleshift_134.jsonl"
    target_path = root / "targets/medcalc_ruleshift_targets_67.jsonl"
    exclusion_path = root / "targets/medcalc_ruleshift_exclusions_13.jsonl"
    write_jsonl(input_path, inputs)
    write_jsonl(target_path, targets)
    write_jsonl(exclusion_path, exclusions)

    audit_path = root / "manual_audit_12.csv"
    audit_rows = []
    for calculator_id in SUPPORTED:
        candidates = [
            target
            for target in targets
            if target["metadata"]["calculator_id"] == calculator_id
        ][:3]
        for target in candidates:
            audit_rows.append(
                {
                    "base_sample_id": target["base_sample_id"],
                    "calculator_id": calculator_id,
                    "criterion": target["metadata"]["criterion_label"],
                    "old_weight": target["metadata"]["old_weight"],
                    "new_weight": target["metadata"]["new_weight"],
                    "original_answer": target["original_answer"],
                    "revised_answer": target["revised_answer"],
                    "automatic_check": "PASS",
                    "optional_human_check": "",
                    "notes": "",
                }
            )
    with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    manifest = {
        "status": "PASS",
        "schema_version": "1.0",
        "build_seed": BUILD_SEED,
        "source": str(source),
        "source_sha256": sha256(source),
        "source_records": len(rows),
        "supported_calculators": SUPPORTED,
        "source_counts": dict(source_by_calculator),
        "retained_cases": len(targets),
        "paired_inputs": len(inputs),
        "excluded_zero_score_cases": len(exclusions),
        "retained_by_calculator": dict(retained_counts),
        "selected_criteria_counts": dict(criteria_counts),
        "revision_delta": DELTA,
        "input_sha256": sha256(input_path),
        "target_sha256": sha256(target_path),
        "exclusion_sha256": sha256(exclusion_path),
        "manual_audit_rows": len(audit_rows),
        "construction_independence": (
            "Selection and revised targets use only frozen reference variables "
            "and official answers; no model prediction is read."
        ),
    }
    manifest_path = root / "task10_input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
