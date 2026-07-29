#!/usr/bin/env python3
"""Local deterministic tests for Task 10 data and evaluator helpers."""

from __future__ import annotations

import importlib.util
import json
import pathlib


def load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(path: pathlib.Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent
    inputs = rows(root / "inputs/medcalc_ruleshift_134.jsonl")
    targets = rows(root / "targets/medcalc_ruleshift_targets_67.jsonl")
    exclusions = rows(
        root / "targets/medcalc_ruleshift_exclusions_13.jsonl"
    )
    assert len(inputs) == 134
    assert len(targets) == 67
    assert len(exclusions) == 13
    assert all(
        set(row) == {"sample_id", "instruction", "metadata"} for row in inputs
    )
    assert all(
        float(row["revised_answer"]) - float(row["original_answer"]) == 2.0
        for row in targets
    )
    assert {
        row["metadata"]["calculator_id"] for row in inputs
    } == {4, 8, 18, 45}

    evaluator = load(root / "evaluate_task10.py", "task10_test_eval")
    assert evaluator.exact_mcnemar([True, False], [False, False])[
        "candidate_only"
    ] == 1
    assert evaluator.paired_bootstrap(
        [1.0, 1.0], [0.0, 0.0], 1
    ) == [1.0, 1.0]
    print("Task 10 local tests PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
