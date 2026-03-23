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
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from _common import _ensure_dir, get_dataset_configs


def _project_root() -> str:
    return os.path.dirname(script_dir)


def _parse_evaluation_txt(path: str) -> dict:
    """Parse evaluation txt; return dict with sil, ch, db, noise_ratio (optional)."""
    out = {"sil": None, "ch": None, "db": None, "noise_ratio": None}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "silhouette" in line.lower() and ":" in line:
                try:
                    out["sil"] = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif "calinski_harabasz" in line.lower() and ":" in line:
                try:
                    out["ch"] = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif "davies_bouldin" in line.lower() and ":" in line:
                try:
                    out["db"] = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif "noise_ratio" in line.lower() and ":" in line:
                try:
                    out["noise_ratio"] = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
    return out


def _plot_spider_comparison(
    results: list[tuple[str, dict, dict]],
    out_path: str,
    title: str,
) -> None:
    """
    Spider/radar chart comparing K-Means vs DBSCAN per dataset.
    Axes (all normalized 0-1, higher=better): Silhouette, CH_norm, 1/(1+DB), Assigned_ratio.
    """
    categories = ["Silhouette", "Calinski-Harabasz\n(normalized)", "Separation\n(1/(1+DB))", "Assigned ratio"]
    n_axes = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    all_ch = []
    for _, km, db in results:
        if km.get("ch") is not None:
            all_ch.append(km["ch"])
        if db.get("ch") is not None:
            all_ch.append(db["ch"])
    ch_max = max(all_ch) if all_ch else 1.0

    fig, axes = plt.subplots(
        1, len(results),
        subplot_kw=dict(projection="polar"),
        figsize=(5 * len(results), 5),
        dpi=160,
        facecolor="#f8f9fa",
    )
    if len(results) == 1:
        axes = [axes]

    for ax, (cfg_name, km, db) in zip(axes, results):
        display_name = cfg_name.replace("_", " ").title()
        ax.set_title(display_name, fontsize=12, fontweight="bold", pad=12)

        def _normalize_metrics(d: dict, is_dbscan: bool) -> list[float]:
            sil = d.get("sil") if d.get("sil") is not None else 0.0
            ch = d.get("ch") if d.get("ch") is not None else 0.0
            db_val = d.get("db") if d.get("db") is not None else 2.0
            sep = 1.0 / (1.0 + db_val)  # higher = better
            ch_norm = ch / ch_max if ch_max > 0 else 0.0
            assigned = 1.0 - d.get("noise_ratio", 0.0) if is_dbscan else 1.0
            return [min(1.0, sil), min(1.0, ch_norm), min(1.0, sep), min(1.0, assigned)]

        km_vals = _normalize_metrics(km, False)
        db_vals = _normalize_metrics(db, True)
        km_vals += km_vals[:1]
        db_vals += db_vals[:1]

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(np.linspace(0, 2 * np.pi, n_axes, endpoint=False)), categories)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
        ax.plot(angles, km_vals, "o-", linewidth=2, label="K-Means", color="#3498db")
        ax.fill(angles, km_vals, alpha=0.25, color="#3498db")
        ax.plot(angles, db_vals, "s-", linewidth=2, label="DBSCAN", color="#e74c3c")
        ax.fill(angles, db_vals, alpha=0.25, color="#e74c3c")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)
        ax.set_facecolor("#f8f9fa")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.08)
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def main() -> None:
    root = _project_root()
    kmeans_dir = os.path.join(root, "webp", "EDA_and_visualization", "07_evaluation", "kmeans")
    dbscan_dir = os.path.join(root, "webp", "EDA_and_visualization", "07_evaluation", "dbscan")
    out_dir = os.path.join(root, "webp", "EDA_and_visualization", "07_evaluation")
    _ensure_dir(out_dir)

    results = []
    results_full = []
    for cfg in get_dataset_configs():
        km_path = os.path.join(kmeans_dir, f"{cfg.name}_evaluation.txt")
        db_path = os.path.join(dbscan_dir, f"{cfg.name}_evaluation.txt")
        if not os.path.exists(km_path) or not os.path.exists(db_path):
            continue
        km_data = _parse_evaluation_txt(km_path)
        db_data = _parse_evaluation_txt(db_path)
        km_sil = km_data.get("sil")
        db_sil = db_data.get("sil")
        if km_sil is not None and db_sil is not None:
            data_src = "DOH" if "distribution_recipient" in cfg.name else "PhilGEPS"
            results.append((cfg.name, km_sil, db_sil, data_src))
            results_full.append((cfg.name, km_data, db_data))

    if len(results) < 1:
        print("No paired K-Means and DBSCAN evaluations found. Run 07_evaluation_kmeans.py and 07_evaluation_dbscan.py first.")
        return

    names = [r[0].replace("_", " ").title() for r in results]
    km_vals = [r[1] for r in results]
    db_vals = [r[2] for r in results]
    x = range(len(names))
    width = 0.35

    # --- Bar chart: Silhouette comparison ---
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

    # --- Spider/radar charts: multi-metric comparison ---
    if results_full:
        _plot_spider_comparison(
            results_full,
            os.path.join(out_dir, "kmeans_vs_dbscan_spider.png"),
            "K-Means vs DBSCAN: Multi-Metric Comparison (higher = better)",
        )
        print(f"Saved spider comparison -> {out_dir}/kmeans_vs_dbscan_spider.png")


if __name__ == "__main__":
    main()
