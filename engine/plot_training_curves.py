"""Plot training curves from batch metric CSV files.

This script reads every results/batch_metrics_*.csv file, normalizes the
training metrics, and writes per-run plus comparison plots to results/plots/.
"""

from __future__ import annotations

import re
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FILENAME_PATTERN = re.compile(
	r"batch_metrics_(?P<model>.+)_(?P<variant>pretrained|scratch)_(?P<timestamp>\d{8}_\d{6})\.csv$"
)


def parse_run_metadata(path: Path) -> dict[str, str]:
	match = FILENAME_PATTERN.search(path.name)
	if not match:
		return {
			"filepath": str(path),
			"run_id": path.stem,
			"model_name": "unknown",
			"variant": "unknown",
			"timestamp": "",
			"label": path.stem,
		}

	model_name = match.group("model")
	variant = match.group("variant")
	timestamp = match.group("timestamp")
	return {
		"filepath": str(path),
		"run_id": f"{model_name}_{variant}_{timestamp}",
		"model_name": model_name,
		"variant": variant,
		"timestamp": timestamp,
		"label": f"{model_name} / {variant}",
	}


def moving_average(values: np.ndarray, window: int = 25) -> np.ndarray:
	if window <= 1 or len(values) < window:
		return values.astype(float)
	kernel = np.ones(window, dtype=float) / window
	return np.convolve(values, kernel, mode="same")


def normalize_run_frame(df: pd.DataFrame) -> pd.DataFrame:
	normalized = df.copy()

	numeric_columns = [
		column
		for column in normalized.columns
		if column not in {"filepath", "run_id", "model_name", "variant", "timestamp", "label"}
	]
	for column in numeric_columns:
		normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

	if "global_step" not in normalized.columns:
		normalized["global_step"] = np.arange(1, len(normalized) + 1)

	normalized["global_step"] = pd.to_numeric(normalized["global_step"], errors="coerce")
	if "epoch" in normalized.columns:
		normalized["epoch"] = pd.to_numeric(normalized["epoch"], errors="coerce")

	for metric in ("batch_loss", "batch_accuracy", "running_loss", "running_accuracy"):
		if metric in normalized.columns:
			normalized[metric] = pd.to_numeric(normalized[metric], errors="coerce")

	normalized = normalized.sort_values("global_step").reset_index(drop=True)
	normalized["loss_smooth"] = moving_average(
		normalized["running_loss"].to_numpy(dtype=float), window=25
	) if "running_loss" in normalized.columns else np.full(len(normalized), np.nan)
	normalized["accuracy_smooth"] = moving_average(
		normalized["running_accuracy"].to_numpy(dtype=float), window=25
	) if "running_accuracy" in normalized.columns else np.full(len(normalized), np.nan)

	return normalized


def load_runs(results_dir: Path) -> dict[str, pd.DataFrame]:
	csv_paths = sorted(Path(path) for path in glob(str(results_dir / "batch_metrics_*.csv")))
	if not csv_paths:
		raise FileNotFoundError(f"No batch_metrics_*.csv files found in {results_dir}")

	runs: dict[str, pd.DataFrame] = {}
	for path in csv_paths:
		metadata = parse_run_metadata(path)
		frame = pd.read_csv(path)
		frame["filepath"] = metadata["filepath"]
		frame["run_id"] = metadata["run_id"]
		frame["model_name"] = metadata["model_name"]
		frame["variant"] = metadata["variant"]
		frame["timestamp"] = metadata["timestamp"]
		frame["label"] = metadata["label"]
		runs[metadata["run_id"]] = normalize_run_frame(frame)

	return runs


def plot_per_run_curves(runs: dict[str, pd.DataFrame], plots_dir: Path) -> list[Path]:
	saved_paths: list[Path] = []

	for run_id, df in runs.items():
		fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 9))

		axes[0].plot(df["global_step"], df["running_loss"], label="running loss", color="tab:blue", linewidth=2)
		if "batch_loss" in df.columns:
			axes[0].plot(df["global_step"], df["batch_loss"], label="batch loss", color="tab:blue", alpha=0.25, linewidth=1)
		axes[0].set_title(f"Training loss - {df['label'].iloc[0]}")
		axes[0].set_ylabel("Loss")
		axes[0].legend(loc="best")
		axes[0].grid(True, alpha=0.3)

		axes[1].plot(df["global_step"], df["running_accuracy"], label="running accuracy", color="tab:green", linewidth=2)
		if "batch_accuracy" in df.columns:
			axes[1].plot(df["global_step"], df["batch_accuracy"], label="batch accuracy", color="tab:green", alpha=0.25, linewidth=1)
		axes[1].set_title(f"Training accuracy - {df['label'].iloc[0]}")
		axes[1].set_xlabel("Global step")
		axes[1].set_ylabel("Accuracy")
		axes[1].legend(loc="best")
		axes[1].grid(True, alpha=0.3)

		fig.tight_layout()
		png_path = plots_dir / f"{run_id}_training_curves.png"
		svg_path = plots_dir / f"{run_id}_training_curves.svg"
		fig.savefig(png_path, dpi=160, bbox_inches="tight")
		fig.savefig(svg_path, bbox_inches="tight")
		plt.close(fig)
		saved_paths.extend([png_path, svg_path])

	return saved_paths


def plot_comparison_curves(runs: dict[str, pd.DataFrame], plots_dir: Path) -> list[Path]:
	fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
	palette = sns.color_palette("tab10", n_colors=max(len(runs), 1))

	for color, (_, df) in zip(palette, runs.items()):
		label = df["label"].iloc[0]
		axes[0].plot(df["global_step"], df["running_loss"], label=label, color=color, linewidth=2)
		axes[1].plot(df["global_step"], df["running_accuracy"], label=label, color=color, linewidth=2)

	axes[0].set_title("Running loss comparison")
	axes[0].set_ylabel("Loss")
	axes[0].legend(loc="best")
	axes[0].grid(True, alpha=0.3)

	axes[1].set_title("Running accuracy comparison")
	axes[1].set_xlabel("Global step")
	axes[1].set_ylabel("Accuracy")
	axes[1].legend(loc="best")
	axes[1].grid(True, alpha=0.3)

	fig.tight_layout()
	png_path = plots_dir / "training_curves_comparison.png"
	svg_path = plots_dir / "training_curves_comparison.svg"
	fig.savefig(png_path, dpi=180, bbox_inches="tight")
	fig.savefig(svg_path, bbox_inches="tight")
	plt.close(fig)
	return [png_path, svg_path]


def main() -> None:
	sns.set_theme(style="whitegrid", context="notebook")
	plt.rcParams.update(
		{
			"figure.figsize": (14, 9),
			"axes.titlesize": 16,
			"axes.labelsize": 12,
			"legend.fontsize": 10,
		}
	)

	project_root = Path(__file__).resolve().parents[1]
	results_dir = project_root / "results"
	plots_dir = results_dir / "plots"
	plots_dir.mkdir(parents=True, exist_ok=True)

	runs = load_runs(results_dir)
	saved_paths = plot_per_run_curves(runs, plots_dir)
	saved_paths.extend(plot_comparison_curves(runs, plots_dir))

	print("Saved plots:")
	for path in saved_paths:
		print(f"- {path}")


if __name__ == "__main__":
	main()
