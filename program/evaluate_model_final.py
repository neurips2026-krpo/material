
import argparse
import datetime as _dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_LLM_API_KEY = "EMPTY"
DEFAULT_LLM_MODEL = "qwen3"
DEFAULT_EMBEDDING_API_URL = "http://127.0.0.1:8000/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


def _clean_optional_text(value: Optional[str]) -> str:
	return (value or "").strip()


def normalize_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
	args.llm_base_url = _clean_optional_text(args.llm_base_url).rstrip("/")
	args.llm_api_key = _clean_optional_text(args.llm_api_key) or DEFAULT_LLM_API_KEY
	args.llm_model = _clean_optional_text(args.llm_model) or DEFAULT_LLM_MODEL
	args.embedding_api_url = _clean_optional_text(args.embedding_api_url).rstrip("/")
	args.embedding_model = _clean_optional_text(args.embedding_model) or DEFAULT_EMBEDDING_MODEL

	if not args.llm_base_url:
		raise ValueError("LLM base URL is required. Provide --llm_base_url or set LLM_BASE_URL.")
	if not args.embedding_api_url:
		raise ValueError(
			"Embedding API URL is required. Provide --embedding_api_url or set EMBEDDING_API_URL."
		)
	return args


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
	items: List[Dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as f:
		for line_no, line in enumerate(f, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except Exception as e:
				raise ValueError(f"Invalid JSONL at {path} line {line_no}: {e}") from e
			items.append(obj)
	return items


class CompatHttpxClient:  
	"""Compatibility shim for OpenAI SDK.

	NOTE: OpenAI Python SDK validates `http_client` via `isinstance(..., httpx.Client)`.
	So this must be (or contain) a real `httpx.Client` instance.
	"""

	def __new__(cls, *args, **kwargs):
		# Lazily import and create a real httpx.Client subclass instance.
		import httpx

		class _CompatHttpxClient(httpx.Client):
			def __init__(self, *a, **kw):
				kw.pop("proxies", None)
				kw.pop("trust_env", None)
				kw.pop("verify", None)
				super().__init__(*a, **kw)

		return _CompatHttpxClient(*args, **kwargs)


@dataclass
class LlmConfig:
	base_url: str
	api_key: str
	model: str
	timeout_s: float = 180.0
	temperature: float = 0.2
	max_tokens: int = 32768


class LlmRunner:
	def __init__(self, cfg: LlmConfig):
		from openai import OpenAI

		try:
			http_client = CompatHttpxClient(timeout=cfg.timeout_s)
			self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, http_client=http_client)
		except Exception:
			self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
		self._cfg = cfg
		self._warned_token_clamp = False

	@staticmethod
	def _model_context_length(model: str) -> int:
		"""Best-effort context length.

		Defaults to 40960 to match common local/OpenAI-compatible deployments.
		Can be overridden via env var `LLM_CONTEXT_LENGTH`.
		"""
		env = os.getenv("LLM_CONTEXT_LENGTH", "").strip()
		if env.isdigit():
			try:
				return int(env)
			except Exception:
				pass

		m = (model or "").lower()
		if "128k" in m or "131072" in m:
			return 131072
		if "64k" in m or "65536" in m:
			return 65536
		if "32k" in m or "32768" in m:
			return 32768
		if "16k" in m or "16384" in m:
			return 16384
		return 40960

	@staticmethod
	def _estimate_input_tokens(messages: List[Dict[str, str]], model: str) -> int:
		"""Roughly estimate tokens for chat messages.

		Uses `tiktoken` if available; otherwise falls back to a heuristic.
		"""
		text = "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in messages)
		try:
			import tiktoken  # type: ignore

			try:
				enc = tiktoken.encoding_for_model(model)
			except Exception:
				enc = tiktoken.get_encoding("cl100k_base")
			# Add a small overhead per message to approximate chat formatting.
			tok = len(enc.encode(text))
			return tok + 8 * max(1, len(messages))
		except Exception:
			return max(1, len(text) // 4)

	def run(self, instruction: str) -> str:
		messages = [{"role": "user", "content": instruction}]
		context_len = self._model_context_length(self._cfg.model)
		input_tokens = self._estimate_input_tokens(messages, self._cfg.model)
		# Keep a safety margin for system/tool overhead and server-side formatting.
		safety = 512
		max_allowed = max(1, context_len - input_tokens - safety)
		max_tokens = min(int(self._cfg.max_tokens), int(max_allowed))
		if max_tokens < int(self._cfg.max_tokens) and not self._warned_token_clamp:
			self._warned_token_clamp = True
			print(
				f"[warn] Clamping max_tokens from {self._cfg.max_tokens} to {max_tokens} "
				f"(context={context_len}, input≈{input_tokens}). Set LLM_CONTEXT_LENGTH to override.",
				file=os.sys.stderr,
			)
		resp = self._client.chat.completions.create(
			messages=messages,
			model=self._cfg.model,
			max_tokens=max_tokens,
			temperature=self._cfg.temperature,
			stream=False,
		)
		return (resp.choices[0].message.content or "").strip()

	def run_with_tokens(self, instruction: str) -> Tuple[str, int]:
		"""Run inference and return both output text and completion tokens count."""
		messages = [{"role": "user", "content": instruction}]
		context_len = self._model_context_length(self._cfg.model)
		input_tokens = self._estimate_input_tokens(messages, self._cfg.model)
		# Keep a safety margin for system/tool overhead and server-side formatting.
		safety = 512
		max_allowed = max(1, context_len - input_tokens - safety)
		max_tokens = min(int(self._cfg.max_tokens), int(max_allowed))
		if max_tokens < int(self._cfg.max_tokens) and not self._warned_token_clamp:
			self._warned_token_clamp = True
			print(
				f"[warn] Clamping max_tokens from {self._cfg.max_tokens} to {max_tokens} "
				f"(context={context_len}, input≈{input_tokens}). Set LLM_CONTEXT_LENGTH to override.",
				file=os.sys.stderr,
			)
		resp = self._client.chat.completions.create(
			messages=messages,
			model=self._cfg.model,
			max_tokens=max_tokens,
			temperature=self._cfg.temperature,
			stream=False,
		)
		output_text = (resp.choices[0].message.content or "").strip()
		# Extract completion tokens from response
		completion_tokens = 0
		if hasattr(resp, 'usage') and resp.usage:
			completion_tokens = getattr(resp.usage, 'completion_tokens', 0)
		return output_text, completion_tokens


class EmbeddingClient:
	"""OpenAI-compatible embedding client (copied/adapted from tcm_reward.py)."""

	def __init__(
		self,
		api_url: Optional[str] = None,
		model: Optional[str] = None,
		timeout: float = 60.0,
	) -> None:
		import httpx

		self.api_url = (api_url or os.getenv("EMBEDDING_API_URL", DEFAULT_EMBEDDING_API_URL)).rstrip("/")
		self.model = model or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
		self._client = httpx.Client(timeout=timeout)

	def embed(self, text: str) -> List[float]:
		payload = {"input": text, "model": self.model}
		try:
			r = self._client.post(self.api_url, json=payload)
			r.raise_for_status()
		except Exception as e:
			raise RuntimeError(f"Embedding API request failed: {e}") from e
		data = r.json()
		try:
			return data["data"][0]["embedding"]
		except Exception as e:
			raise RuntimeError(f"Invalid embedding response: {data}") from e


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
	"""Extract the first JSON object { ... } found in text."""
	try:
		text = text.strip()
		match = re.search(r"\{.*\}", text, flags=re.DOTALL)
		if not match:
			return None
		return json.loads(match.group())
	except Exception:
		return None


def _extract_actions(text: str) -> Dict[str, str]:
	"""Return mapping: action_index(str) -> raw_content(str)."""
	# Handles both:
	# - action2:[总体计划]: {...}
	# - action2:{...}
	action_pat = re.compile(r"^action(\d+):(\s*(?:\[[^\]]*\]:\s*)?.*)$", flags=re.MULTILINE)
	actions = {}
	for m in action_pat.finditer(text):
		idx = m.group(1)
		content = m.group(2).strip()
		# If it was action2:[xxx]: {...}, strip the [xxx]: prefix.
		content = re.sub(r"^\[[^\]]*\]:\s*", "", content)
		actions[idx] = content
	return actions


def _format_ok(text: str) -> bool:
	# Keep consistent with reward: require <reasoning>...<analysis>... and </reasoning>
	return bool(re.search(r"<reasoning>.*?<analysis>.*?</reasoning>", text, flags=re.DOTALL))


class SimilarityScorer:
	"""Embedding-based semantic similarity with the same non-linear mapping as tcm_reward.py."""

	def __init__(self, embedding_client: EmbeddingClient):
		import torch

		self._client = embedding_client
		self._embed_cache: Dict[str, "torch.Tensor"] = {}

	def _get_embed(self, text: str):
		import torch

		if len(self._embed_cache) > 5000:
			self._embed_cache.clear()
		if text not in self._embed_cache:
			self._embed_cache[text] = torch.tensor(self._client.embed(text)).to(torch.float32)
		return self._embed_cache[text]

	def semantic_similarity(self, str1: str, str2: str) -> float:
		import torch.nn.functional as F

		if not str1 or not str2:
			return 0.0
		s1, s2 = str1.strip(), str2.strip()
		if s1 == s2:
			return 1.0

		try:
			v1 = self._get_embed(s1)
			v2 = self._get_embed(s2)
		except Exception:
			return 0.0

		raw_sim = float(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item())

		if raw_sim >= 1.0:
			return 1.0
		if raw_sim < 0.72:
			return raw_sim * 0.1
		return float(raw_sim ** 2.0)


def _score_json_objects(
	sim: SimilarityScorer,
	pred_json: Optional[Dict[str, Any]],
	gt_json: Optional[Dict[str, Any]],
	expected_keys: Optional[List[str]] = None,
	exclude_keys: Optional[List[str]] = None,
) -> float:
	if not pred_json or not gt_json:
		return 0.0

	format_score = 1.0
	if expected_keys is not None:
		if set(pred_json.keys()) != set(expected_keys):
			format_score -= 0.3

	for length_key, limit in [("综合结论", 40)]:
		if length_key in pred_json and len(str(pred_json[length_key])) > limit:
			format_score -= 0.2

	# Exclude specified keys from semantic similarity scoring
	if exclude_keys is None:
		exclude_keys = []
	compare_keys = [k for k in gt_json.keys() if k not in exclude_keys]
	scores: List[float] = []
	for k in compare_keys:
		if k not in pred_json:
			scores.append(0.0)
			continue
		scores.append(sim.semantic_similarity(str(pred_json[k]), str(gt_json[k])))
	base = sum(scores) / len(scores) if scores else 0.0
	return max(0.0, base * format_score)


def _normalize_pred_json_keys_for_gt(pred_json: Dict[str, Any], gt_json: Dict[str, Any]) -> Dict[str, Any]:
	"""Normalize common key variants so scoring compares the intended fields.

	This evaluator primarily scores action modules by comparing keys present in GT.
	In some datasets/models, the same semantic field is emitted as:
	- "辨证结果"  vs "分析结果"
	To avoid unfair 0-scores (missing key) and the -0.3 format penalty (key set mismatch),
	we rename Pred keys to match GT when possible.
	"""
	# Work on a copy to avoid surprising callers.
	p = dict(pred_json)

	# If GT expects 辨证结果 but Pred used 分析结果, rename.
	if "辨证结果" in gt_json and "辨证结果" not in p and "分析结果" in p:
		p["辨证结果"] = p.pop("分析结果")
	# If GT expects 分析结果 but Pred used 辨证结果, rename.
	if "分析结果" in gt_json and "分析结果" not in p and "辨证结果" in p:
		p["分析结果"] = p.pop("辨证结果")

	# If Pred contains both variants but GT contains only one, drop the extra to match key set.
	if "辨证结果" in gt_json and "分析结果" in p and "分析结果" not in gt_json:
		p.pop("分析结果", None)
	if "分析结果" in gt_json and "辨证结果" in p and "辨证结果" not in gt_json:
		p.pop("辨证结果", None)

	return p


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


class BagangEvaluator:
	def __init__(self, sim: SimilarityScorer):
		self._sim = sim

	def score_pair(self, pred: str, gt: str, sample_id: int) -> SampleScore:
		ok = _format_ok(pred)
		if not ok:
			return SampleScore(
				sample_id=sample_id,
				overall_0_100=0.0,
				reward_0_1=0.0,
				format_ok=False,
				action_avg_0_1=0.0,
				summary_0_1=0.0,
				per_action_0_1={},
				missing_actions=[],
			)

		pred_actions = _extract_actions(pred)
		gt_actions = _extract_actions(gt)

		per_action: Dict[str, float] = {}
		missing: List[str] = []
		action_scores: List[float] = []

		for idx, gt_content in gt_actions.items():
			if idx == "1":
				continue

			pred_content = pred_actions.get(idx)
			if pred_content is None:
				missing.append(idx)
				per_action[idx] = 0.0
				action_scores.append(0.0)
				continue

			p_json = _extract_first_json_object(pred_content)
			g_json = _extract_first_json_object(gt_content)
			if p_json and g_json:
				p_json = _normalize_pred_json_keys_for_gt(p_json, g_json)
				expected = list(g_json.keys())
				# For action modules, exclude "简短原因" and "下一步骤" from scoring
				a_score = _score_json_objects(self._sim, p_json, g_json, expected_keys=expected, exclude_keys=["简短原因", "下一步骤"])

				# Same special-case bonus for concatenated "辨证结果" (e.g. action7)
				gt_res = g_json.get("辨证结果") or g_json.get("分析结果")
				pred_res = p_json.get("辨证结果") or p_json.get("分析结果")
				if isinstance(gt_res, str) and "/" in gt_res:
					if isinstance(pred_res, str) and "/" in pred_res:
						if len(pred_res.split("/")) == len(gt_res.split("/")):
							a_score = min(1.0, a_score + 0.1)
				per_action[idx] = a_score
				action_scores.append(a_score)
			else:
				# fallback: text-level similarity
				a_score = self._sim.semantic_similarity(pred_content, gt_content)
				per_action[idx] = a_score
				action_scores.append(a_score)

		action_avg = (sum(action_scores) / len(action_scores)) if action_scores else 0.0

		# Summary scoring: compare JSON after </reasoning>
		pred_summary_part = pred.split("</reasoning>")[-1]
		gt_summary_part = gt.split("</reasoning>")[-1]
		p_sum_json = _extract_first_json_object(pred_summary_part)
		g_sum_json = _extract_first_json_object(gt_summary_part)
		if p_sum_json and g_sum_json:
			# For summary module, exclude "综合结论" from scoring
			summary_score = _score_json_objects(self._sim, p_sum_json, g_sum_json, expected_keys=list(g_sum_json.keys()), exclude_keys=["综合结论"])
		else:
			summary_score = self._sim.semantic_similarity(pred_summary_part, gt_summary_part)

		# Final weighted reward (0-1), same weights as tcm_reward.py
		reward = 0.1 + 0.5 * action_avg + 0.4 * summary_score
		reward = max(0.0, min(1.0, reward))
		return SampleScore(
			sample_id=sample_id,
			overall_0_100=reward * 100.0,
			reward_0_1=reward,
			format_ok=True,
			action_avg_0_1=action_avg,
			summary_0_1=summary_score,
			per_action_0_1=per_action,
			missing_actions=missing,
		)


def _safe_write_text(path: Path, text: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(text, encoding="utf-8")


def _truncate_for_excel(text: str, limit: int = 30000) -> str:
	if text is None:
		return ""
	text = str(text)
	if len(text) <= limit:
		return text
	return text[:limit] + "\n...[TRUNCATED]"


def _detect_category(instruction: str) -> str:
	kws = {
		"八纲": "八纲辨证",
		"病因综合辨证": "病因辨证",
		"六经辨证": "六经辨证",
		"气血津液辨证": "气血津液辨证",
		"三焦辨证": "三焦辨证",
		"卫气营血辨证": "卫气营血辨证",
		"脏腑辨证": "脏腑辨证",
	}
	for kw, cat in kws.items():
		if kw in instruction:
			return cat
	return "综合/其他"


def export_excel(rows: List[Dict[str, Any]], out_xlsx: Path) -> None:
	import pandas as pd

	out_xlsx.parent.mkdir(parents=True, exist_ok=True)
	df = pd.DataFrame(rows)

	def _get_summary(d):
		return {
			"n": len(d),
			"avg": float(d["score_0_100"].mean()) if len(d) else 0.0,
			"p50": float(d["score_0_100"].median()) if len(d) else 0.0,
			"min": float(d["score_0_100"].min()) if len(d) else 0.0,
			"max": float(d["score_0_100"].max()) if len(d) else 0.0,
		}

	summary_all = _get_summary(df)
	df_summary_all = pd.DataFrame([summary_all])

	with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
		df.to_excel(writer, index=False, sheet_name="results_all")
		df_summary_all.to_excel(writer, index=False, sheet_name="summary_all")

		# Add category summary sheet
		if "category" in df.columns:
			cats = sorted(df["category"].unique())
			category_summary_rows = []
			for cat in cats:
				df_cat = df[df["category"] == cat]
				category_summary_rows.append({
					"辨证类型": cat,
					"样本数": len(df_cat),
					"平均分": round(float(df_cat["score_0_100"].mean()), 2) if len(df_cat) else 0.0,
					"中位数": round(float(df_cat["score_0_100"].median()), 2) if len(df_cat) else 0.0,
					"最低分": round(float(df_cat["score_0_100"].min()), 2) if len(df_cat) else 0.0,
					"最高分": round(float(df_cat["score_0_100"].max()), 2) if len(df_cat) else 0.0,
				})
			# Add overall summary row
			category_summary_rows.append({
				"辨证类型": "总体",
				"样本数": len(df),
				"平均分": round(float(df["score_0_100"].mean()), 2) if len(df) else 0.0,
				"中位数": round(float(df["score_0_100"].median()), 2) if len(df) else 0.0,
				"最低分": round(float(df["score_0_100"].min()), 2) if len(df) else 0.0,
				"最高分": round(float(df["score_0_100"].max()), 2) if len(df) else 0.0,
			})
			df_category_summary = pd.DataFrame(category_summary_rows)
			df_category_summary.to_excel(writer, index=False, sheet_name="辨证类型汇总")

		# Add sheets for each category
		if "category" in df.columns:
			cats = sorted(df["category"].unique())
			for cat in cats:
				df_cat = df[df["category"] == cat]
				# Excel sheet names limited to 31 chars
				safe_name = str(cat)[:30]
				df_cat.to_excel(writer, index=False, sheet_name=safe_name)
				cat_sum = _get_summary(df_cat)
				pd.DataFrame([cat_sum]).to_excel(writer, index=False, sheet_name=f"{safe_name}_sum")


def export_report(rows: List[Dict[str, Any]], out_dir: Path) -> None:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import pandas as pd

	out_dir.mkdir(parents=True, exist_ok=True)
	df = pd.DataFrame(rows)
	scores = df["score_0_100"].astype(float).tolist() if len(df) else []

	# 1) Histogram
	fig = plt.figure(figsize=(8, 4.5))
	plt.hist(scores, bins=20, range=(0, 100))
	plt.title("Score distribution (0-100)")
	plt.xlabel("score")
	plt.ylabel("count")
	hist_path = out_dir / "score_hist.png"
	plt.tight_layout()
	plt.savefig(hist_path, dpi=180)
	plt.close(fig)

	# 2) Per-action mean bar (sorted dynamically)
	action_cols = [c for c in df.columns if c.startswith("action_") and c.endswith("_score")]
	action_means = {c: float(df[c].mean()) for c in action_cols} if action_cols else {}
	fig = plt.figure(figsize=(10, 4.5))

	def _extract_num(s: str) -> int:
		match = re.search(r"(\d+)", s)
		return int(match.group(1)) if match else 0

	labels = sorted(action_means.keys(), key=_extract_num)
	values = [action_means[k] for k in labels]
	plt.bar(labels, values)
	plt.ylim(0, 1)
	plt.title("Per-action average (0-1)")
	plt.ylabel("avg similarity")
	plt.xticks(rotation=45, ha="right")
	bar_path = out_dir / "action_avg.png"
	plt.tight_layout()
	plt.savefig(bar_path, dpi=180)
	plt.close(fig)

	# 3) Category mean bar
	cat_means = {}
	if "category" in df.columns:
		cat_means = df.groupby("category")["score_0_100"].mean().to_dict()

	if cat_means:
		fig = plt.figure(figsize=(10, 4.5))
		cat_labels = sorted(cat_means.keys())
		cat_values = [cat_means[k] for k in cat_labels]
		plt.bar(cat_labels, cat_values, color="orange")
		plt.ylim(0, 100)
		plt.title("Average Score by Diagnosis Type (0-100)")
		plt.ylabel("avg score")
		plt.xticks(rotation=45, ha="right")
		cat_bar_path = out_dir / "category_avg.png"
		plt.tight_layout()
		plt.savefig(cat_bar_path, dpi=180)
		plt.close(fig)

	# HTML index
	avg_score = float(df["score_0_100"].mean()) if len(df) else 0.0
	cat_rows_html = ""
	if cat_means:
		cat_rows_html = "<h3>各辨证类型分析</h3><table><tr><th>辨证类型</th><th>样本数</th><th>平均分</th></tr>"
		counts = df["category"].value_counts().to_dict()
		for cat in sorted(cat_means.keys()):
			cat_rows_html += f"<tr><td>{cat}</td><td>{counts[cat]}</td><td>{cat_means[cat]:.2f}</td></tr>"
		cat_rows_html += "</table>"

	html = f"""<!doctype html>
<html lang=\"zh\">
<head>
  <meta charset=\"utf-8\" />
  <title>TCM LLM Evaluation Report</title>
  <style>
	body {{ font-family: system-ui, -apple-system, Segoe UI, Arial, sans-serif; margin: 24px; }}
	.row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
	.card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; }}
	img {{ max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 6px; margin-bottom: 20px; }}
	table {{ border-collapse: collapse; width: 100%; max-width: 600px; margin-bottom: 24px; }}
	th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
	th {{ background-color: #f5f5f5; }}
  </style>
</head>
<body>
  <h2>TCM LLM 推理评估报告</h2>
  <div class=\"row\">
	<div class=\"card\">总样本数: {len(df)}</div>
	<div class=\"card\">总体平均分(0-100): {avg_score:.2f}</div>
  </div>
  
  <h3>分数分布 (总体)</h3>
  <img src=\"score_hist.png\" alt=\"score histogram\" />
  
  {'<h3>辨证维度平均分</h3><img src="category_avg.png" alt="category averages" />' if cat_means else ''}
  {cat_rows_html}

  <h3>Action 维度平均相似度 (总体)</h3>
  <img src=\"action_avg.png\" alt=\"action averages\" />
</body>
</html>"""
	(out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Evaluate LLM reasoning outputs with semantic similarity scoring.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"--data",
		type=str,
		default=str(Path(__file__).parent / "tcm_test_datasets.jsonl"),
		help="Path to jsonl test data ({instruction, output}).",
	)
	parser.add_argument("--max_samples", type=int, default=0, help="If >0, only evaluate first N samples.")
	parser.add_argument("--out_dir", type=str, default="", help="Output directory (default: ./outputs/<timestamp>).")
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
		out_dir = Path(__file__).parent / "outputs" / ts
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
	evaluator = BagangEvaluator(sim)

	try:
		from tqdm import tqdm

		iterator: Iterable[Tuple[int, Dict[str, Any]]] = tqdm(list(enumerate(items, start=1)))
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
			"missing_actions": ",".join(score.missing_actions),
			"pred_path": str(pred_path),
			"gt_path": str(gt_path),
			"instruction_trunc": _truncate_for_excel(instruction, limit=2000),
			"response": _truncate_for_excel(pred),
		}

		# Dynamic action columns (normalized to 0-1)
		for action_idx, act_score in score.per_action_0_1.items():
			row[f"action_{action_idx}_score"] = round(float(act_score), 6)

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
