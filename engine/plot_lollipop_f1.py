import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_lollipop(df, title, save_path, x_col='class_name', y_col='f1_score'):
    """
    Creates a vertical lollipop chart (stem plot) for the given dataframe.
    """
    plt.figure(figsize=(14, 6))
    
    # Ensure it's sorted by class name if needed (optional)
    # df = df.sort_values(by=x_col)
    
    x_pos = range(len(df))
    
    # Create the vertical lines (the "sticks")
    plt.vlines(x=x_pos, ymin=0, ymax=df[y_col], color='skyblue', alpha=0.8, linewidth=4)
    
    # Create the top circles (the "lollipops")
    plt.plot(x_pos, df[y_col], "o", markersize=8, color='navy')
    
    # Formatting
    plt.ylim(0, 1.05)
    plt.title(title, fontsize=16)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.xticks(x_pos, df[x_col], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

def main():
    # Paths to the input CSVs
    results_dir = Path("results/testing")
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Process the 26-class multi-model CSV
    f1_csv = results_dir / "per_class_f1.csv"
    if f1_csv.exists():
        df = pd.read_csv(f1_csv)
        # Group by the 'run_key' or 'model' since there are multiple evaluated models
        for run_key, group in df.groupby('run_key'):
            model_name = group['model'].iloc[0]
            save_path = plot_dir / f"{run_key}_f1_lollipop.png"
            plot_lollipop(group, f"Per-class F1 Score: {model_name}", save_path)
    else:
        print(f"File not found: {f1_csv}")
        
    # 2. Process the 29-class CSV
    f1_29_csv = results_dir / "per_class_f1_29.csv"
    if f1_29_csv.exists():
        df_29 = pd.read_csv(f1_29_csv)
        save_path_29 = plot_dir / "mobilenet_v2_finetuned_29_f1_lollipop.png"
        plot_lollipop(df_29, "Per-class F1 Score: MobileNetV2 Finetuned (29 classes)", save_path_29)
    else:
        print(f"File not found: {f1_29_csv}")

if __name__ == "__main__":
    main()
