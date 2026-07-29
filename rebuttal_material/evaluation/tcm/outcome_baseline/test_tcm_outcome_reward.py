#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import re
import tempfile

from tcm_outcome_reward import compute_score


def read_first(path: pathlib.Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def replace_final_field(text: str, field: str, value: str) -> str:
    head, tail = text.split("</reasoning>", maxsplit=1)
    objects = []
    decoder = json.JSONDecoder()
    for index, character in enumerate(tail):
        if character != "{":
            continue
        try:
            candidate, end = decoder.raw_decode(tail[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            objects.append((index, index + end, candidate))
    start, end, summary = objects[-1]
    summary[field] = value
    return head + "</reasoning>" + tail[:start] + json.dumps(
        summary, ensure_ascii=False, separators=(",", ":")
    ) + tail[end:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=pathlib.Path, default=None)
    args = parser.parse_args()
    workspace = pathlib.Path(__file__).resolve().parents[3]
    train_file = args.train_file or (
        workspace / "project/EasyR1/datasets/tcm_train_2nd_datasets.jsonl"
    )
    row = read_first(train_file)
    target = row["output"]

    perfect = compute_score({"response": target, "ground_truth": target})
    assert perfect == {"overall": 1.0, "format": 1.0, "accuracy": 1.0}

    action_changed = re.sub(
        r'(?m)^(action\d+[^\n]*"分析结果"\s*:\s*")[^"]*',
        r"\1故意修改的中间过程",
        target,
        count=1,
    )
    assert action_changed != target
    assert compute_score(
        {"response": action_changed, "ground_truth": target}
    ) == perfect

    conclusion_changed = replace_final_field(target, "综合结论", "故意修改")
    assert compute_score(
        {"response": conclusion_changed, "ground_truth": target}
    ) == perfect

    summary_match = re.search(
        r'\{[^{}]*"病因"[^{}]*\}\s*$', target, flags=re.DOTALL
    )
    assert summary_match is not None
    summary = json.loads(summary_match.group(0))
    summary["病因"] = "故意错误"
    wrong_final = target[: summary_match.start()] + json.dumps(
        summary, ensure_ascii=False, separators=(",", ":")
    )
    wrong_score = compute_score(
        {"response": wrong_final, "ground_truth": target}
    )
    assert wrong_score["format"] == 1.0
    assert 0.0 < wrong_score["overall"] < 1.0

    no_closing_tag = target.replace("</reasoning>", "", 1)
    invalid = compute_score(
        {"response": no_closing_tag, "ground_truth": target}
    )
    assert invalid == {"overall": 0.0, "format": 0.0, "accuracy": 0.0}

    with tempfile.TemporaryDirectory() as temporary:
        marker = pathlib.Path(temporary) / "PASS"
        marker.write_text("PASS\n", encoding="utf-8")
        assert marker.read_text(encoding="utf-8") == "PASS\n"
    print("Task 9 outcome-only reward tests PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
