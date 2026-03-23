"""
7. Evaluation - K-Means vs DBSCAN Comparison (CRISP-DM)

Creates combined comparison charts from K-Means and DBSCAN evaluation outputs.
Inputs: webp/EDA_and_visualization/07_evaluation/kmeans/*.txt, dbscan/*.txt
Outputs: webp/EDA_and_visualization/07_evaluation/
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from _common import _ensure_dir, get_dataset_configs


def _project_root() -> str:
    return os.path.dirname(script_dir)


def main() -> None:
    root = _project_root()
    kmeans_dir = os.path.join(root, "webp", "EDA_and_visualization", "07_evaluation", "kmeans")
    dbscan_dir = os.path.join(root, "webp", "EDA_and_visualization", "07_evaluation", "dbscan")
    out_dir = os.path.join(root, "webp", "EDA_and_visualization", "07_evaluation")
    _ensure_dir(out_dir)

    results = []
    for cfg in get_dataset_configs():
        km_path = os.path.join(kmeans_dir, f"{cfg.name}_evaluation.txt")
        db_path = os.path.join(dbscan_dir, f"{cfg.name}_evaluation.txt")
        if not os.path.exists(km_path) or not os.path.exists(db_path):
            continue
        km_sil = db_sil = None
        for path in (km_path, db_path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if "silhouette" in line.lower() and ":" in line:
                        val = float(line.split(":")[1].strip())
                        if path == km_path:
                            km_sil = val
                        else:
                            db_sil = val
                        break
        if km_sil is not None and db_sil is not None:
            data_src = "DOH" if "distribution_recipient" in cfg.name else "PhilGEPS"
            results.append((cfg.name, km_sil, db_sil, data_src))

    if len(results) < 1:
        print("No paired K-Means and DBSCAN evaluations found. Run 07_evaluation_kmeans.py and 07_evaluation_dbscan.py first.")
        return

    names = [r[0].replace("_", " ").title() for r in results]
    km_vals = [r[1] for r in results]
    db_vals = [r[2] for r in results]
    x = range(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160, facecolor="#f8f9fa")
    bars1 = ax.bar([i - width / 2 for i in x], km_vals, width, label="K-Means", color="#3498db", edgecolor="#2c3e50")
    bars2 = ax.bar([i + width / 2 for i in x], db_vals, width, label="DBSCAN", color="#e74c3c", edgecolor="#2c3e50")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Silhouette Score (higher = better)", fontsize=12)
    ax.set_title("K-Means vs DBSCAN: Silhouette Score Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_facecolor("#f8f9fa")
    for b in bars1:
        ax.text(b.get_x() + b.get_width() / 2 - 0.02, b.get_height() + 0.02, f"{b.get_height():.2f}", ha="center", fontsize=8, fontweight="bold")
    for b in bars2:
        ax.text(b.get_x() + b.get_width() / 2 + 0.02, b.get_height() + 0.02, f"{b.get_height():.2f}", ha="center", fontsize=8, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kmeans_vs_dbscan_silhouette.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Saved K-Means vs DBSCAN comparison -> {out_dir}/kmeans_vs_dbscan_silhouette.png")


if __name__ == "__main__":
    main()
