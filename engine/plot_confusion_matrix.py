"""
plot_confusion_matrix.py
------------------------
Reads one or more confusion-matrix CSV files and plots each one as a
normalised heatmap.

Each CSV must have:
  - A first column with the true class labels (used as the row index).
  - One column per predicted class (column headers = predicted labels).

The script normalises each row by its sum so values represent the
fraction of true-class samples predicted as each class (i.e. recall
per cell).  The diagonal therefore shows per-class recall.

Usage
-----
    python plot_confusion_matrix.py <csv1> [<csv2> ...] [options]

Examples
--------
    # Plot all four models interactively
    python plot_confusion_matrix.py *_confusion_matrix.csv

    # Save to PNG (one file per model, named after the input CSV)
    python plot_confusion_matrix.py *_confusion_matrix.csv --save

    # Custom colourmap and DPI
    python plot_confusion_matrix.py *_confusion_matrix.csv --save --cmap viridis --dpi 200

Arguments
---------
    csv_files   One or more paths to confusion-matrix CSV files (positional).
    --save      Save each figure as <stem>.png instead of showing interactively.
    --dpi N     Resolution for saved figures (default: 150).
    --cmap C    Matplotlib colormap name (default: Blues).
    --no-annot  Suppress per-cell text annotations (useful for large matrices).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_confusion_matrix(csv_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load a CSV confusion matrix.  First column = row labels (true classes)."""
    df = pd.read_csv(csv_path, index_col=0)
    classes = list(df.index)
    cm = df.values.astype(float)
    return cm, classes


def normalise_rows(cm: np.ndarray) -> np.ndarray:
    """Normalise each row to sum to 1 (recall-normalised matrix)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    return cm / row_sums


def pretty_title(stem: str) -> str:
    """Turn a file stem like 'inception_v3_scratch' into a readable title."""
    return stem.replace("_confusion_matrix", "").replace("_", " ").title()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_single_cm(
    cm_norm: np.ndarray,
    cm_raw: np.ndarray,
    classes: list[str],
    title: str,
    cmap: str = "Blues",
    annotate: bool = True,
) -> plt.Figure:
    n = len(classes)
    cell_size = max(0.52, 10 / n)
    fig_w = n * cell_size + 2.5
    fig_h = n * cell_size + 1.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap,
                   norm=Normalize(vmin=0, vmax=1), aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of true-class samples", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    if annotate:
        font_size = max(4, 9 - n // 8)
        thresh = 0.5  # switch text colour above this normalised value
        for i in range(n):
            for j in range(n):
                val_norm = cm_norm[i, j]
                val_raw = int(cm_raw[i, j])
                if val_raw == 0:
                    continue
                color = "white" if val_norm > thresh else "black"
                ax.text(
                    j, i,
                    f"{val_norm:.2f}\n({val_raw})",
                    ha="center", va="center",
                    color=color,
                    fontsize=font_size,
                    linespacing=1.3,
                )

    # Overall accuracy in a subtitle
    correct = np.trace(cm_raw)
    total = cm_raw.sum()
    acc = correct / total if total > 0 else 0.0
    ax.set_xlabel(
        f"Predicted label          (overall accuracy = {acc:.4f}  |  "
        f"{int(correct)}/{int(total)} samples)",
        fontsize=10,
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot confusion matrices from CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "csv_files", nargs="+", type=Path,
        help="One or more confusion-matrix CSV files.",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save figures to PNG instead of displaying interactively.",
    )
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for saved figures (default: 150).")
    parser.add_argument("--cmap", type=str, default="Blues",
                        help="Matplotlib colormap (default: Blues).")
    parser.add_argument("--no-annot", action="store_true",
                        help="Disable per-cell text annotations.")
    args = parser.parse_args(argv)

    missing = [p for p in args.csv_files if not p.exists()]
    if missing:
        sys.exit(f"Error: file(s) not found: {missing}")

    for csv_path in args.csv_files:
        print(f"Processing: {csv_path.name}")
        cm_raw, classes = load_confusion_matrix(csv_path)
        cm_norm = normalise_rows(cm_raw)
        title = pretty_title(csv_path.stem)

        fig = plot_single_cm(
            cm_norm, cm_raw, classes,
            title=title,
            cmap=args.cmap,
            annotate=not args.no_annot,
        )

        if args.save:
            out_path = Path(csv_path.stem).with_suffix(".png")
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            print(f"  Saved → {out_path}")
            plt.close(fig)
        else:
            plt.show()

    if not args.save:
        print("All figures displayed.")


if __name__ == "__main__":
    main()
