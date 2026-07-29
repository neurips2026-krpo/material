#!/usr/bin/env python3
"""Build deterministic Task 8 incomplete/random/conflicting TCM prompts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import random
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable


SEED = 20260726
LEVELS = (75, 50, 25, 0)
EXPECTED = {
    "clean": "4ed28b82ffd56364e2c598a51f032ef55cfaa7287e6f186b9675c07e313b4fe7",
    "counterfactual": "6b6fd88f5c065a885d5da4f411410edeef659aebd9d069eaa66a0d17ed46f00b",
    "targets": "63c52792e1b2ea3f8865d25c87f6571c18924f62a033b607833bc234d92c5cc6",
}
ACTION_PATTERN = re.compile(r"(?i)action\s*(\d+)")
SYMPTOM_BLOCK = re.compile(
    r"(【症状知识库】\n)(.*?)(?=\n\n【|\n\n请)", re.S
)
RULE_LINE = re.compile(r"(?m)^-\s+([^：:\n]+)[：:].*$")
NUMBERED = re.compile(r"(?m)^\s*\d+[.)、]\s*")


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    result: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def parse_action_labels(text: str) -> list[str]:
    markers = list(ACTION_PATTERN.finditer(text))
    labels: list[str] = []
    for offset, marker in enumerate(markers):
        if int(marker.group(1)) <= 2:
            continue
        end = (
            markers[offset + 1].start()
            if offset + 1 < len(markers)
            else len(text)
        )
        objects = json_objects(text[marker.end() : end])
        if not objects:
            continue
        for field in ("辨证结果", "分析结果", "子类"):
            value = objects[0].get(field)
            if isinstance(value, str) and value.strip() and value != "无":
                labels.append(value)
    return labels


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("淤", "瘀")
    return re.sub(
        r"[\s（）()\[\]【】“”\"'‘’、，,;；/:：_]+", "", value
    )


def components(value: str) -> list[str]:
    return [
        normalized(part)
        for part in re.split(r"[-—/]", value)
        if normalized(part) and normalized(part) != "无"
    ]


def split_numbered(text: str) -> list[str]:
    markers = list(NUMBERED.finditer(text))
    return [
        text[
            marker.end() : (
                markers[index + 1].start()
                if index + 1 < len(markers)
                else len(text)
            )
        ].strip()
        for index, marker in enumerate(markers)
        if text[
            marker.end() : (
                markers[index + 1].start()
                if index + 1 < len(markers)
                else len(text)
            )
        ].strip()
    ]


def parse_symptoms(instruction: str) -> tuple[list[str], re.Match[str]]:
    match = SYMPTOM_BLOCK.search(instruction)
    if match is None:
        raise ValueError("missing symptom knowledge block")
    value = json.loads(match.group(2))
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError("clean symptom knowledge must be a JSON string list")
    return value, match


def parse_cf_units(instruction: str) -> list[str]:
    match = SYMPTOM_BLOCK.search(instruction)
    if match is None:
        raise ValueError("missing counterfactual symptom knowledge")
    units = split_numbered(match.group(2).strip())
    for block in re.finditer(
        r"【[^】]*反事实[^】]*】\n(.*?)(?=\n\n【|\n\n请)",
        instruction,
        re.S,
    ):
        units.extend(split_numbered(block.group(1)))
    result: list[str] = []
    for unit in units:
        if unit and unit not in result:
            result.append(unit)
    if len(result) < 5:
        raise ValueError(
            f"counterfactual prompt has only {len(result)} knowledge units"
        )
    return result[:5]


def target_components(target: dict[str, Any]) -> list[str]:
    labels = parse_action_labels(str(target["clean_output"]))
    labels.extend(
        value
        for field, value in target["final_summaries"]["clean"].items()
        if field != "综合结论" and isinstance(value, str)
    )
    result: list[str] = []
    for label in labels:
        result.extend(components(label))
    return list(dict.fromkeys(result))


def seeded_order(
    values: list[str], sample_id: str, mode: str
) -> list[str]:
    payload = f"{SEED}:{sample_id}:{mode}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    result = list(values)
    random.Random(seed).shuffle(result)
    return result


def knowledge_inventory(
    instruction: str,
    reference_components: list[str],
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    symptoms, _ = parse_symptoms(instruction)
    units: list[dict[str, str]] = [
        {"unit_id": f"S{index:03d}", "kind": "symptom", "text": text}
        for index, text in enumerate(symptoms, start=1)
    ]
    matched_rules: list[str] = []
    for index, match in enumerate(RULE_LINE.finditer(instruction), start=1):
        label = match.group(1).strip()
        unit_id = f"R{index:03d}"
        units.append(
            {
                "unit_id": unit_id,
                "kind": "diagnostic_rule",
                "label": label,
                "text": match.group(0),
            }
        )
        label_parts = components(label)
        leaf = label_parts[-1] if label_parts else ""
        if any(
            (leaf == target or leaf in target or target in leaf)
            and min(len(leaf), len(target)) >= 2
            for target in reference_components
        ):
            matched_rules.append(unit_id)
    relevant = [
        unit["unit_id"]
        for unit in units
        if unit["kind"] == "symptom"
    ][:4] + matched_rules
    return units, relevant, matched_rules


def mutate_instruction(
    instruction: str,
    units: list[dict[str, str]],
    delete_ids: set[str],
    inject_units: list[str],
) -> str:
    symptoms, match = parse_symptoms(instruction)
    symptom_ids = {
        f"S{index:03d}" for index in range(1, len(symptoms) + 1)
    }
    kept_symptoms = [
        text
        for index, text in enumerate(symptoms, start=1)
        if f"S{index:03d}" not in delete_ids
    ]
    kept_symptoms.extend(inject_units)
    replacement = match.group(1) + json.dumps(
        kept_symptoms, ensure_ascii=False, indent=2
    )
    result = instruction[: match.start()] + replacement + instruction[match.end() :]
    rule_text_by_id = {
        unit["unit_id"]: unit["text"]
        for unit in units
        if unit["kind"] == "diagnostic_rule"
    }
    deleted_rule_text = {
        rule_text_by_id[unit_id]
        for unit_id in delete_ids
        if unit_id not in symptom_ids and unit_id in rule_text_by_id
    }
    result = "\n".join(
        line
        for line in result.splitlines()
        if line not in deleted_rule_text
    )
    return result


def deletion_count(n_relevant: int, quality: int) -> int:
    return math.ceil(n_relevant * (1 - quality / 100))


def conflict_count(quality: int) -> int:
    return math.floor(5 * (1 - quality / 100) + 0.5)


def normalized_final(value: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (field, normalized(str(label)))
            for field, label in value.items()
            if field != "综合结论"
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=pathlib.Path, required=True)
    parser.add_argument("--counterfactual", type=pathlib.Path, required=True)
    parser.add_argument("--targets", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    paths = {
        "clean": args.clean.resolve(),
        "counterfactual": args.counterfactual.resolve(),
        "targets": args.targets.resolve(),
    }
    for name, path in paths.items():
        observed = sha256(path)
        if observed != EXPECTED[name]:
            raise ValueError(f"{name} hash mismatch: {observed}")

    clean_rows = read_jsonl(paths["clean"])
    cf_rows = {row["sample_id"]: row for row in read_jsonl(paths["counterfactual"])}
    targets = {row["sample_id"]: row for row in read_jsonl(paths["targets"])}
    if (
        len(clean_rows) != 100
        or len(cf_rows) != 100
        or len(targets) != 100
    ):
        raise ValueError("Task 8 requires 100 paired source cases")
    if {row["sample_id"] for row in clean_rows} != set(cf_rows) or set(cf_rows) != set(targets):
        raise ValueError("source sample IDs do not align")

    output_root = args.output_root.resolve()
    inputs_root = output_root / "inputs"
    generated: dict[str, list[dict[str, Any]]] = {
        f"{mode}_q{quality:03d}": []
        for mode in ("relevant", "random", "conflict")
        for quality in LEVELS
    }
    maps: list[dict[str, Any]] = []
    audit_candidates: dict[str, list[dict[str, str]]] = {
        mode: [] for mode in ("relevant", "random", "conflict")
    }

    for source in clean_rows:
        sample_id = source["sample_id"]
        reference = target_components(targets[sample_id])
        units, relevant_ids, matched_rule_ids = knowledge_inventory(
            source["instruction"], reference
        )
        all_ids = [unit["unit_id"] for unit in units]
        relevant_order = seeded_order(relevant_ids, sample_id, "relevant")
        random_order = seeded_order(all_ids, sample_id, "random")
        cf_units = parse_cf_units(cf_rows[sample_id]["instruction"])
        unit_by_id = {unit["unit_id"]: unit for unit in units}
        eligible = bool(relevant_ids)
        clean_final = targets[sample_id]["final_summaries"]["clean"]
        fixed_final = targets[sample_id]["final_summaries"][
            "counterfactual_fixed"
        ]
        conflict_eligible = (
            normalized_final(clean_final) != normalized_final(fixed_final)
        )
        map_row = {
            "sample_id": sample_id,
            "task_family": source["metadata"]["task_family"],
            "primary_leakage_free": source["metadata"]["primary_leakage_free"],
            "eligible_relevant_deletion": eligible,
            "eligible_conflict": conflict_eligible,
            "reference_components": reference,
            "knowledge_units": units,
            "relevant_unit_ids": relevant_ids,
            "matched_rule_ids": matched_rule_ids,
            "relevant_deletion_order": relevant_order,
            "random_deletion_order": random_order,
            "counterfactual_units": cf_units,
        }
        maps.append(map_row)

        for quality in LEVELS:
            count = deletion_count(len(relevant_ids), quality)
            deletion_sets = {
                "relevant": relevant_order[:count],
                "random": random_order[:count],
            }
            for mode, deleted in deletion_sets.items():
                instruction = mutate_instruction(
                    source["instruction"], units, set(deleted), []
                )
                condition = f"{mode}_q{quality:03d}"
                metadata = {
                    **source["metadata"],
                    "task8_mode": mode,
                    "nominal_quality": quality,
                    "eligible_relevant_deletion": eligible,
                    "knowledge_units_total": len(units),
                    "relevant_units_total": len(relevant_ids),
                    "deleted_unit_ids": deleted,
                    "deleted_units_count": len(deleted),
                    "source_clean_instruction_sha256": sha256_text(
                        source["instruction"]
                    ),
                    "task8_instruction_sha256": sha256_text(instruction),
                }
                generated[condition].append(
                    {
                        "sample_id": sample_id,
                        "instruction": instruction,
                        "metadata": metadata,
                    }
                )
                if eligible:
                    audit_candidates[mode].append(
                        {
                            "mode": mode,
                            "quality": str(quality),
                            "sample_id": sample_id,
                            "task_family": source["metadata"]["task_family"],
                            "eligible": str(eligible),
                            "changed": str(
                                instruction != source["instruction"]
                            ),
                            "deleted_unit_ids": "|".join(deleted),
                            "deleted_text": "\n---\n".join(
                                unit_by_id[item]["text"] for item in deleted
                            ),
                            "injected_text": "",
                            "instruction_sha256": sha256_text(instruction),
                            "_source_instruction": source["instruction"],
                            "_task8_instruction": instruction,
                            "_reference_components": json.dumps(
                                reference, ensure_ascii=False
                            ),
                        }
                    )

            injected_count = conflict_count(quality)
            injected = cf_units[:injected_count]
            instruction = mutate_instruction(
                source["instruction"], units, set(), injected
            )
            condition = f"conflict_q{quality:03d}"
            generated[condition].append(
                {
                    "sample_id": sample_id,
                    "instruction": instruction,
                    "metadata": {
                        **source["metadata"],
                        "task8_mode": "conflict",
                        "nominal_quality": quality,
                        "eligible_conflict": conflict_eligible,
                        "injected_conflict_units": injected_count,
                        "source_clean_instruction_sha256": sha256_text(
                            source["instruction"]
                        ),
                        "task8_instruction_sha256": sha256_text(instruction),
                    },
                }
            )
            if conflict_eligible:
                audit_candidates["conflict"].append(
                    {
                        "mode": "conflict",
                        "quality": str(quality),
                        "sample_id": sample_id,
                        "task_family": source["metadata"]["task_family"],
                        "eligible": "True",
                        "changed": str(instruction != source["instruction"]),
                        "deleted_unit_ids": "",
                        "deleted_text": "",
                        "injected_text": "\n---\n".join(injected),
                        "instruction_sha256": sha256_text(instruction),
                        "_source_instruction": source["instruction"],
                        "_task8_instruction": instruction,
                        "_reference_components": json.dumps(
                            reference, ensure_ascii=False
                        ),
                    }
                )

    files: dict[str, dict[str, Any]] = {}
    for condition, rows in generated.items():
        path = inputs_root / f"tcm_{condition}_100.jsonl"
        write_jsonl(path, rows)
        files[condition] = {
            "relative_path": str(path.relative_to(output_root)),
            "records": len(rows),
            "sha256": sha256(path),
        }
    map_path = output_root / "relevant_rule_map_100.jsonl"
    write_jsonl(map_path, maps)

    audit_rows: list[dict[str, str]] = []
    audit_packet_root = output_root / "manual_audit_packet"
    audit_packet_root.mkdir(parents=True, exist_ok=True)
    for old_packet in audit_packet_root.glob("*.md"):
        old_packet.unlink()
    for mode, candidates in audit_candidates.items():
        selected: list[dict[str, str]] = []
        by_family: dict[str, list[dict[str, str]]] = {}
        for row in candidates:
            by_family.setdefault(row["task_family"], []).append(row)
        for family in sorted(by_family):
            ordered = sorted(
                by_family[family],
                key=lambda row: sha256_text(
                    f"{SEED}:{mode}:{row['sample_id']}:{row['quality']}"
                ),
            )
            selected.extend(ordered[:2])
        remaining = [
            row for row in candidates if row not in selected
        ]
        remaining.sort(
            key=lambda row: sha256_text(
                f"{SEED}:extra:{mode}:{row['sample_id']}:{row['quality']}"
            )
        )
        selected.extend(remaining[: 20 - len(selected)])
        for row in selected:
            packet_name = (
                f"{row['mode']}_{row['sample_id']}_q"
                f"{int(row['quality']):03d}.md"
            )
            packet_path = audit_packet_root / packet_name
            packet_path.write_text(
                "\n".join(
                    [
                        f"# {row['mode']} / {row['sample_id']} / "
                        f"quality={row['quality']}",
                        "",
                        "## Frozen clean reference components",
                        "",
                        row["_reference_components"],
                        "",
                        "## Deleted knowledge units",
                        "",
                        row["deleted_text"] or "(none)",
                        "",
                        "## Injected conflicting units",
                        "",
                        row["injected_text"] or "(none)",
                        "",
                        "## Original answer-free prompt",
                        "",
                        row["_source_instruction"],
                        "",
                        "## Perturbed answer-free prompt",
                        "",
                        row["_task8_instruction"],
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            visible_row = {
                key: value for key, value in row.items() if not key.startswith("_")
            }
            audit_rows.append(
                {
                    **visible_row,
                    "audit_file": str(
                        packet_path.relative_to(output_root)
                    ),
                    "human_perturbation_valid_yes_no": "",
                    "human_no_answer_leakage_yes_no": "",
                    "human_prompt_grammatical_yes_no": "",
                    "human_notes": "",
                }
            )
    audit_path = output_root / "manual_audit_60.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    eligible = [row for row in maps if row["eligible_relevant_deletion"]]
    conflict_eligible = [row for row in maps if row["eligible_conflict"]]
    manifest = {
        "status": "PASS",
        "protocol_frozen_before_model_outputs": True,
        "seed": SEED,
        "levels_generated": list(LEVELS),
        "clean_level_reused_from_task6": 100,
        "task6_no_kg_external_anchor": True,
        "task6_counterfactual_extreme_anchor": True,
        "source_hashes": {
            name: sha256(path) for name, path in paths.items()
        },
        "records": 100,
        "primary_leakage_free_records": sum(
            bool(row["metadata"]["primary_leakage_free"])
            for row in clean_rows
        ),
        "relevant_deletion_eligible_records": len(eligible),
        "relevant_deletion_eligible_primary_records": sum(
            bool(row["primary_leakage_free"]) for row in eligible
        ),
        "ineligible_sample_ids": [
            row["sample_id"]
            for row in maps
            if not row["eligible_relevant_deletion"]
        ],
        "conflict_eligible_records": len(conflict_eligible),
        "conflict_eligible_primary_records": sum(
            bool(row["primary_leakage_free"]) for row in conflict_eligible
        ),
        "conflict_ineligible_sample_ids": [
            row["sample_id"] for row in maps if not row["eligible_conflict"]
        ],
        "family_counts": dict(
            Counter(row["metadata"]["task_family"] for row in clean_rows)
        ),
        "generated_input_files": files,
        "relevant_rule_map": {
            "relative_path": str(map_path.relative_to(output_root)),
            "records": len(maps),
            "sha256": sha256(map_path),
        },
        "manual_audit": {
            "relative_path": str(audit_path.relative_to(output_root)),
            "records": len(audit_rows),
            "per_mode": dict(Counter(row["mode"] for row in audit_rows)),
            "sha256": sha256(audit_path),
            "status": "PENDING_HUMAN_COMPLETION",
            "packet_files": len(list(audit_packet_root.glob("*.md"))),
        },
    }
    manifest_path = output_root / "task8_input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
