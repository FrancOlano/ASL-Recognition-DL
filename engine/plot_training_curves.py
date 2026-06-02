from __future__ import annotations

import json
import re
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

FILENAME_PATTERN = re.compile(
    r"history_(?P<model>.+)_(?P<variant>pretrained|scratch)_(?P<timestamp>\d{8}_\d{6})\.json$"
)

def parse_run_metadata(path: Path) -> dict:
    match = FILENAME_PATTERN.search(path.name)
    if not match:
        return {
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
        "run_id": f"{model_name}_{variant}_{timestamp}",
        "model_name": model_name,
        "variant": variant,
        "timestamp": timestamp,
        "label": f"{model_name} / {variant}",
    }

def load_runs(results_dir: Path) -> dict:
    json_paths = sorted(Path(path) for path in glob(str(results_dir / "history_*.json")))
    if not json_paths:
        raise FileNotFoundError(f"No history_*.json files found in {results_dir}")

    runs = {}
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        metadata = parse_run_metadata(path)
        data.update(metadata)
        runs[metadata["run_id"]] = data

    return runs

def plot_per_run_curves(runs: dict, plots_dir: Path) -> list[Path]:
    saved_paths = []

    for run_id, data in runs.items():
        epochs = range(1, len(data.get("train_losses", [])) + 1)
        if not epochs:
            continue
            
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 9))

        axes[0].plot(epochs, data["train_losses"], label="Train Loss", color="tab:blue", linewidth=2)
        if "val_losses" in data:
            axes[0].plot(epochs, data["val_losses"], label="Validation Loss", color="tab:orange", linewidth=2)
        axes[0].set_title(f"Loss - {data['label']}")
        axes[0].set_ylabel("Loss")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(epochs)

        axes[1].plot(epochs, data["train_accuracies"], label="Train Accuracy", color="tab:green", linewidth=2)
        if "val_accuracies" in data:
            axes[1].plot(epochs, data["val_accuracies"], label="Validation Accuracy", color="tab:red", linewidth=2)
        axes[1].set_title(f"Accuracy - {data['label']}")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(epochs)

        fig.tight_layout()
        png_path = plots_dir / f"{run_id}_curves.png"
        fig.savefig(png_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(png_path)

    return saved_paths

def plot_comparison_curves(runs: dict, plots_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
    palette = sns.color_palette("tab10", n_colors=max(len(runs), 1))

    for color, (_, data) in zip(palette, runs.items()):
        epochs = range(1, len(data.get("val_accuracies", data.get("train_accuracies", []))) + 1)
        if not epochs:
            continue
            
        label = data["label"]
        val_losses = data.get("val_losses", data.get("train_losses"))
        val_accuracies = data.get("val_accuracies", data.get("train_accuracies"))
        
        axes[0].plot(epochs, val_losses, label=label, color=color, linewidth=2, marker='o')
        axes[1].plot(epochs, val_accuracies, label=label, color=color, linewidth=2, marker='o')

    axes[0].set_title("Validation Loss Comparison")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation Accuracy Comparison")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = plots_dir / "curves_comparison.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [png_path]

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
