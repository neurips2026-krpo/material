import argparse
import datetime as _dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from evaluate_model_final import (
	DEFAULT_EMBEDDING_API_URL,
	DEFAULT_EMBEDDING_MODEL,
	DEFAULT_LLM_API_KEY,
	DEFAULT_LLM_BASE_URL,
	DEFAULT_LLM_MODEL,
	EmbeddingClient,
	LlmConfig,
	LlmRunner,
	SimilarityScorer,
	_detect_category,
	normalize_runtime_args,
	_safe_write_text,
	_score_json_objects,
	_truncate_for_excel,
	export_excel,
	export_report,
	read_jsonl,
)


def _extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
	"""Extract the last valid JSON object found in text.

	Used for no-reasoning outputs where the final answer is a single JSON object,
	optionally prefixed by text like "八纲辨证推理分析总结：".
	"""
	text = (text or "").strip()
	if not text:
		return None

	decoder = json.JSONDecoder()
	objects: List[Dict[str, Any]] = []
	for i, ch in enumerate(text):
		if ch != "{":
			continue
		try:
			obj, _ = decoder.raw_decode(text[i:])
		except Exception:
			continue
		if isinstance(obj, dict):
			objects.append(obj)
	return objects[-1] if objects else None


@dataclass
class SampleScore:
	sample_id: int
	overall_0_100: float
	reward_0_1: float
	format_ok: bool
	action_avg_0_1: float
	summary_0_1: float
	per_action_0_1: Dict[str, float]
	missing_actions: List[str]


class FinalOnlyEvaluator:
	"""Evaluate only the final summary result JSON (no reasoning/action scoring)."""

	def __init__(self, sim: SimilarityScorer):
		self._sim = sim

	def score_pair(self, pred: str, gt: str, sample_id: int) -> SampleScore:
		p_sum_json = _extract_last_json_object(pred)
		g_sum_json = _extract_last_json_object(gt)

		if p_sum_json and g_sum_json:
			summary_score = _score_json_objects(
				self._sim,
				p_sum_json,
				g_sum_json,
				expected_keys=list(g_sum_json.keys()),
				exclude_keys=["综合结论"],
			)
			format_ok = True
		else:
			# Fallback for malformed/non-JSON outputs
			summary_score = self._sim.semantic_similarity(pred, gt)
			format_ok = bool(p_sum_json)

		reward = max(0.0, min(1.0, summary_score))
		return SampleScore(
			sample_id=sample_id,
			overall_0_100=reward * 100.0,
			reward_0_1=reward,
			format_ok=format_ok,
			action_avg_0_1=0.0,
			summary_0_1=summary_score,
			per_action_0_1={},
			missing_actions=[],
		)


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Evaluate LLM outputs by scoring only the final summary JSON (no reasoning).",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"--data",
		type=str,
		default=str(Path(__file__).parent / "tcm_test_no_reasoning_datasets.jsonl"),
		help="Path to jsonl test data ({instruction, output}).",
	)
	parser.add_argument("--max_samples", type=int, default=0, help="If >0, only evaluate first N samples.")
	parser.add_argument("--out_dir", type=str, default="", help="Output directory (default: ./outputs/<timestamp>_no_reasoning).")
	parser.add_argument(
		"--llm_base_url",
		type=str,
		default=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
		help="OpenAI-compatible chat completion endpoint.",
	)
	parser.add_argument(
		"--llm_api_key",
		type=str,
		default=os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY),
		help="API key for the chat model service. Use a placeholder such as EMPTY for local no-auth servers.",
	)
	parser.add_argument("--llm_model", type=str, default=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL))
	parser.add_argument(
		"--embedding_api_url",
		type=str,
		default=os.getenv("EMBEDDING_API_URL", DEFAULT_EMBEDDING_API_URL),
		help="OpenAI-compatible embedding endpoint.",
	)
	parser.add_argument(
		"--embedding_model",
		type=str,
		default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
	)
	args = normalize_runtime_args(parser.parse_args())

	data_path = Path(args.data)
	if not data_path.exists():
		raise FileNotFoundError(f"Data file not found: {data_path}")

	if args.out_dir:
		out_dir = Path(args.out_dir)
	else:
		ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
		out_dir = Path(__file__).parent / "outputs" / f"{ts}_no_reasoning"
	out_dir.mkdir(parents=True, exist_ok=True)

	items = read_jsonl(data_path)
	if args.max_samples and args.max_samples > 0:
		items = items[: args.max_samples]

	llm = LlmRunner(
		LlmConfig(
			base_url=args.llm_base_url,
			api_key=args.llm_api_key,
			model=args.llm_model,
		)
	)

	emb_client = EmbeddingClient(
		api_url=args.embedding_api_url or None,
		model=args.embedding_model or None,
	)
	sim = SimilarityScorer(emb_client)
	evaluator = FinalOnlyEvaluator(sim)

	try:
		from tqdm import tqdm

		iterator: Iterable[Any] = tqdm(list(enumerate(items, start=1)))
	except Exception:
		iterator = enumerate(items, start=1)

	rows: List[Dict[str, Any]] = []
	preds_dir = out_dir / "predictions"
	gts_dir = out_dir / "ground_truth"

	for i, item in iterator:
		instruction = item.get("instruction", "")
		gt = item.get("output", "")
		pred, output_tokens = llm.run_with_tokens(instruction)

		category = _detect_category(instruction)

		pred_path = preds_dir / f"{i:04d}.txt"
		gt_path = gts_dir / f"{i:04d}.txt"
		_safe_write_text(pred_path, pred)
		_safe_write_text(gt_path, gt)

		score = evaluator.score_pair(pred, gt, sample_id=i)

		row: Dict[str, Any] = {
			"id": i,
			"category": category,
			"score_0_100": round(score.overall_0_100, 4),
			"reward_0_1": round(score.reward_0_1, 6),
			"format_ok": score.format_ok,
			"action_avg_0_1": round(score.action_avg_0_1, 6),
			"summary_0_1": round(score.summary_0_1, 6),
			"output_tokens": output_tokens,
			"missing_actions": "",
			"pred_path": str(pred_path),
			"gt_path": str(gt_path),
			"instruction_trunc": _truncate_for_excel(instruction, limit=2000),
			"response": _truncate_for_excel(pred),
		}

		rows.append(row)

	# Exports
	export_excel(rows, out_dir / "results.xlsx")
	export_report(rows, out_dir / "report")
	(out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

	avg = sum(r["score_0_100"] for r in rows) / len(rows) if rows else 0.0
	print(f"Done. n={len(rows)} avg_score_0_100={avg:.2f} out_dir={out_dir}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
