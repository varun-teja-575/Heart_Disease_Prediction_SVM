from pathlib import Path
import matplotlib.pyplot as plt

def plot_target_distribution(df, target_col, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    counts = df[target_col].value_counts().sort_index()
    plt.figure(figsize=(6, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Heart Disease Distribution")
    plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
