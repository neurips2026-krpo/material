import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="从 Excel 的 results_all sheet 计算准确率并导出结果。"
	)
	parser.add_argument(
		"-i",
		"--input",
		required=True,
		help="输入 Excel 文件路径，例如 results3.xlsx",
	)
	parser.add_argument(
		"-o",
		"--output",
		default="",
		help="输出 Excel 文件路径；不传则默认 <输入文件名>_acc_original.xlsx",
	)
	return parser.parse_args()


def resolve_output_path(input_path: Path, output_arg: str) -> Path:
	if output_arg:
		return Path(output_arg)
	return input_path.with_name(f"{input_path.stem}_acc_original.xlsx")


def load_results_all(input_path: Path) -> pd.DataFrame:
	try:
		df = pd.read_excel(input_path, sheet_name="results_all")
	except ValueError as exc:
		xls = pd.ExcelFile(input_path)
		sheets = ", ".join(xls.sheet_names)
		raise ValueError(
			f"未找到 sheet 'results_all'。当前可用 sheet: {sheets}"
		) from exc

	if "score_0_100" not in df.columns:
		raise ValueError("sheet 'results_all' 中缺少字段 'score_0_100'")

	return df


def build_output_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	work_df = df.copy()

	work_df["score_0_100"] = pd.to_numeric(work_df["score_0_100"], errors="coerce")
	work_df["is_correct_75"] = work_df["score_0_100"] >= 75
	work_df["is_strict_correct_80"] = work_df["score_0_100"] >= 80

	total = len(work_df)
	valid = int(work_df["score_0_100"].notna().sum())
	correct_75 = int(work_df["is_correct_75"].sum())
	correct_80 = int(work_df["is_strict_correct_80"].sum())

	acc_75 = (correct_75 / valid) if valid else 0.0
	acc_80 = (correct_80 / valid) if valid else 0.0

	summary = pd.DataFrame(
		[
			{"metric": "total_rows", "value": total},
			{"metric": "valid_score_rows", "value": valid},
			{"metric": "correct_count_ge_75", "value": correct_75},
			{"metric": "accuracy_ge_75", "value": acc_75},
			{"metric": "strict_correct_count_ge_80", "value": correct_80},
			{"metric": "strict_accuracy_ge_80", "value": acc_80},
		]
	)

	return work_df, summary


def main() -> int:
	args = parse_args()
	input_path = Path(args.input)
	if not input_path.exists():
		raise FileNotFoundError(f"输入文件不存在: {input_path}")

	output_path = resolve_output_path(input_path, args.output)
	df = load_results_all(input_path)
	detail_df, summary_df = build_output_frames(df)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
		detail_df.to_excel(writer, index=False, sheet_name="results_all_acc")
		summary_df.to_excel(writer, index=False, sheet_name="acc_summary")

	acc_75 = float(summary_df.loc[summary_df["metric"] == "accuracy_ge_75", "value"].iloc[0])
	acc_80 = float(summary_df.loc[summary_df["metric"] == "strict_accuracy_ge_80", "value"].iloc[0])

	print(f"输入文件: {input_path}")
	print(f"输出文件: {output_path}")
	print(f"准确率(>=75): {acc_75:.4%}")
	print(f"严格准确率(>=80): {acc_80:.4%}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
