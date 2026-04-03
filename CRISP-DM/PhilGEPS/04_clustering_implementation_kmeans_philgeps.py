"""
Step 04a — PhilGEPS k-means.

Same search/fit pattern as the DOH script, but iterates get_philgeps_dataset_configs() (A–G).
Large tables (notably lens B) may switch to MiniBatchKMeans inside the implementation to keep
runtimes practical.

Outputs: this_datasets/PhilGEPS/04_clustering/*_kmeans.csv + figures under
webp/EDA_and_visualization/PhilGEPS/04_clustering_implementation/kmeans/.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Shared `_common`, `sources_paths`, etc. live in parent `CRISP-DM/`.
script_dir = os.path.dirname(os.path.abspath(__file__))
_crisp_dm_root = os.path.dirname(script_dir)
if _crisp_dm_root not in sys.path:
    sys.path.insert(0, _crisp_dm_root)

from _common import (
    _add_data_source_badge,
    _ensure_dir,
    _get_data_source_colors,
    FIG_RECT_WITH_FOOTER,
    data_source_for_dataset_name,
    get_philgeps_dataset_configs,
    load_features,
)
from log_tee import tee_stdio_to_file
from sources_paths import data_root_philgeps, logs_dir_philgeps, webp_root_philgeps


def fit_kmeans(X: np.ndarray, k: int, *, random_state: int = 42, use_minibatch: bool = False):
    # Use MiniBatchKMeans only for large datasets to keep runtime reasonable.
    if use_minibatch:
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=3, batch_size=1024)
    else:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return km, labels


def plot_silhouette_vs_k(
    silhouette_by_k: dict[int, float],
    best_k: int,
    out_path: str,
    data_source: str | None,
) -> None:
    """Plot silhouette score vs k for parameter choice documentation."""
    if not silhouette_by_k:
        return
    _ensure_dir(os.path.dirname(out_path))
    colors = _get_data_source_colors(data_source)
    ks = sorted(silhouette_by_k.keys())
    vals = [silhouette_by_k[k] for k in ks]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160, facecolor=colors["bg"])
    ax.plot(ks, vals, "o-", color=colors["primary"], linewidth=2.5, markersize=10)
    ax.axvline(x=best_k, color=colors["secondary"], linestyle="--", linewidth=2, alpha=0.9, label=f"Selected k={best_k}")
    ax.set_xlabel("k (number of clusters)", fontsize=11)
    ax.set_ylabel("Silhouette score (higher = better)", fontsize=11)
    ax.set_title("K-Means: Silhouette vs k", fontsize=12, fontweight="bold")
    y_lo = max(-0.05, min(vals) - 0.05)
    y_hi = min(1.05, max(vals) + 0.05)
    if y_lo >= y_hi:
        y_hi = y_lo + 0.1
    ax.set_ylim(y_lo, y_hi)
    ax.set_facecolor(colors["bg"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout(rect=FIG_RECT_WITH_FOOTER)
    if data_source:
        _add_data_source_badge(ax, data_source)
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def plot_hyperparam_tuning_silhouette_and_clusters(
    *,
    silhouette_by_k: dict[int, float],
    clusters_by_k: dict[int, int],
    best_k: int,
    out_path: str,
    data_source: str | None,
) -> None:
    """Thesis-friendly hyperparameter tuning plot for K-Means (silhouette + number of clusters)."""
    if not silhouette_by_k:
        return
    _ensure_dir(os.path.dirname(out_path))
    colors = _get_data_source_colors(data_source)

    ks = sorted(silhouette_by_k.keys())
    sil_vals = [silhouette_by_k[k] for k in ks]
    cluster_vals = [clusters_by_k.get(k, 0) for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160, facecolor=colors["bg"])

    # Left: silhouette vs k
    ax = axes[0]
    ax.plot(ks, sil_vals, "o-", color=colors["primary"], linewidth=2.5, markersize=10)
    ax.axvline(x=best_k, color=colors["secondary"], linestyle="--", linewidth=2, alpha=0.9)
    ax.set_title(f"Silhouette vs k (selected k={best_k})", fontsize=11, fontweight="bold")
    ax.set_xlabel("k (clusters)", fontsize=10)
    ax.set_ylabel("Silhouette score", fontsize=10)
    ax.set_facecolor(colors["bg"])
    ax.grid(True, alpha=0.3)

    # Right: number of clusters vs k
    ax = axes[1]
    ax.plot(ks, cluster_vals, "s-", color=colors["secondary"], linewidth=2.5, markersize=10)
    ax.axvline(x=best_k, color=colors["primary"], linestyle="--", linewidth=2, alpha=0.9)
    ax.set_title("Number of clusters vs k", fontsize=11, fontweight="bold")
    ax.set_xlabel("k (clusters)", fontsize=10)
    ax.set_ylabel("n_clusters", fontsize=10)
    ax.set_facecolor(colors["bg"])
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"K-Means Hyperparameter Tuning - {data_source or 'N/A'}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def plot_kmeans_clustering_results_by_k(
    *,
    X_2d: np.ndarray,
    labels_by_k: dict[int, np.ndarray],
    silhouette_by_k: dict[int, float],
    best_k: int,
    out_path: str,
    data_source: str | None,
) -> None:
    """Grid of PCA(2D) clustering results across candidate k values."""
    if not labels_by_k:
        return

    _ensure_dir(os.path.dirname(out_path))
    colors = _get_data_source_colors(data_source)
    ks = sorted(labels_by_k.keys())
    n = len(ks)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows), dpi=160, facecolor=colors["bg"])
    if n_rows == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, k in enumerate(ks):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        labels = labels_by_k[k]
        sil = silhouette_by_k.get(k, 0.0)
        is_best = k == best_k

        sc = ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=labels,
            cmap="tab10",
            alpha=0.75,
            s=10,
            edgecolors="white",
            linewidth=0.25,
        )
        ax.set_title(f"k={k}" + (" (best)" if is_best else "") + f"\nSil={sil:.3f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_facecolor(colors["bg"])
        ax.grid(True, alpha=0.2)

        if is_best:
            for spine in ax.spines.values():
                spine.set_linewidth(2.2)
                spine.set_color(colors["primary"])

    # Hide any unused subplots
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    fig.suptitle(f"K-Means clustering results across candidate k (PCA 2D) - {data_source or 'N/A'}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def main() -> None:
    cluster_dir = os.path.join(data_root_philgeps(), "04_clustering")
    out_dir = os.path.join(webp_root_philgeps(), "04_clustering_implementation", "kmeans")
    _ensure_dir(cluster_dir)
    _ensure_dir(out_dir)
    term_log = os.path.join(logs_dir_philgeps(), "04_clustering_implementation_kmeans_philgeps_terminal.txt")
    with tee_stdio_to_file(term_log):
        _run_kmeans_philgeps(cluster_dir, out_dir)


def _run_kmeans_philgeps(cluster_dir: str, out_dir: str) -> None:
    sweep_by_dataset: dict[str, dict[str, object]] = {}

    for cfg in get_philgeps_dataset_configs():
        result = load_features(cfg)
        if result is None:
            print(f"K-Means [{cfg.name}]: Skip (file missing or empty)")
            continue

        df, _, X = result
        n_samples = X.shape[0]
        # Keep k search bounded so we don't request more clusters than the data can support.
        k_max = min(10, n_samples // 10, 8)
        k_range_actual = range(2, max(3, k_max + 1))
        use_minibatch = n_samples > 5000
        data_src = data_source_for_dataset_name(cfg.name)

        # --- Step 2: Sweep k; choose k with best Silhouette score ---
        silhouette_by_k: dict[int, float] = {}
        clusters_by_k: dict[int, int] = {}
        labels_by_k: dict[int, np.ndarray] = {}
        best_k = 3
        best_sil = -1.0
        # For very large datasets, silhouette computation is expensive; sample points for speed.
        sil_sample = min(5000, n_samples) if n_samples > 5000 else None
        for k in k_range_actual:
            if k >= n_samples:
                break
            _, labels = fit_kmeans(X, k, use_minibatch=use_minibatch)
            if len(np.unique(labels)) < 2:
                continue
            n_clusters = int(len(np.unique(labels)))
            if sil_sample and n_samples > sil_sample:
                rng = np.random.default_rng(42)
                idx = rng.choice(n_samples, sil_sample, replace=False)
                sil = float(silhouette_score(X[idx], labels[idx]))
            else:
                sil = float(silhouette_score(X, labels))
            silhouette_by_k[k] = sil
            clusters_by_k[k] = n_clusters
            labels_by_k[k] = labels
            # Pick k with the highest silhouette score over the evaluated range.
            if sil > best_sil:
                best_sil = sil
                best_k = k

        plot_path = os.path.join(out_dir, f"{cfg.name}_silhouette_vs_k.png")
        plot_silhouette_vs_k(silhouette_by_k, best_k, plot_path, data_src)
        print(f"K-Means [{cfg.name}]: Saved silhouette vs k -> {plot_path}")

        # --- Step 2b: Visualize clustering results across candidate k (PCA 2D grid) ---
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
        # PCA projection is purely for visualization; clustering is still performed in original scaled space.
        tuning_results_path = os.path.join(out_dir, f"{cfg.name}_clustering_results_by_k.png")
        plot_kmeans_clustering_results_by_k(
            X_2d=X_2d,
            labels_by_k=labels_by_k,
            silhouette_by_k=silhouette_by_k,
            best_k=best_k,
            out_path=tuning_results_path,
            data_source=data_src,
        )
        print(f"K-Means [{cfg.name}]: Saved clustering results by k -> {tuning_results_path}")

        tuning_path = os.path.join(out_dir, f"{cfg.name}_hyperparam_tuning.png")
        plot_hyperparam_tuning_silhouette_and_clusters(
            silhouette_by_k=silhouette_by_k,
            clusters_by_k=clusters_by_k,
            best_k=best_k,
            out_path=tuning_path,
            data_source=data_src,
        )
        print(f"K-Means [{cfg.name}]: Saved hyperparameter tuning -> {tuning_path}")

        # --- Step 3: Fit final K-Means and assign cluster labels ---
        km_final, labels_final = fit_kmeans(X, best_k, use_minibatch=False)
        df_out = df.copy()
        df_out["cluster_kmeans"] = labels_final
        out_csv = os.path.join(cluster_dir, f"clustering_{cfg.name}_kmeans.csv")
        df_out.to_csv(out_csv, index=False)

        n_clusters = len(np.unique(labels_final))
        if n_clusters >= 2:
            # Final silhouette is computed on a sample for large n, mirroring the sweep logic.
            if n_samples > 5000:
                rng = np.random.default_rng(42)
                idx = rng.choice(n_samples, min(5000, n_samples), replace=False)
                sil_final = silhouette_score(X[idx], labels_final[idx])
            else:
                sil_final = silhouette_score(X, labels_final)
        else:
            sil_final = 0.0

        print(f"K-Means [{cfg.name}]: k={best_k}, clusters={n_clusters}, silhouette={sil_final:.4f} -> {out_csv}")
        with open(os.path.join(out_dir, f"{cfg.name}_params.txt"), "w", encoding="utf-8") as f:
            f.write(f"k={best_k}\nmethod=sil_vs_k\ninertia={km_final.inertia_:.4f}\nsilhouette={sil_final:.4f}\noutput={out_csv}\n")

        sweep_by_dataset[cfg.name] = {
            "silhouette_by_k": silhouette_by_k,
            "clusters_by_k": clusters_by_k,
            "best_k": best_k,
            "data_src": data_src,
        }

    # --- Combined hyperparameter tuning visualization (A/B/C in one figure) ---
    # Layout: 2 rows x 3 columns
    # Row 1: silhouette vs k
    # Row 2: n_clusters vs k
    if sweep_by_dataset:
        colors = _get_data_source_colors(None)
        cfg_names = [c.name for c in get_philgeps_dataset_configs() if c.name in sweep_by_dataset]
        ncols = max(1, len(cfg_names))
        fig, axes = plt.subplots(2, ncols, figsize=(5.5 * ncols, 10), dpi=160, facecolor=colors["bg"])
        if ncols == 1:
            axes = np.asarray([[axes[0]], [axes[1]]])
        for col, cfg_name in enumerate(cfg_names):
            payload = sweep_by_dataset[cfg_name]
            silhouette_by_k = payload["silhouette_by_k"]
            clusters_by_k = payload["clusters_by_k"]
            best_k = int(payload["best_k"])
            data_src = payload["data_src"]

            ks = sorted(silhouette_by_k.keys())  # type: ignore[union-attr]
            sil_vals = [silhouette_by_k[k] for k in ks]  # type: ignore[index]
            cluster_vals = [clusters_by_k.get(k, 0) for k in ks]  # type: ignore[union-attr]

            # Row 1: silhouette
            ax = axes[0, col]
            ax.plot(ks, sil_vals, "o-", color=colors["secondary"], linewidth=2.5, markersize=8)
            ax.axvline(x=best_k, color=colors["primary"], linestyle="--", linewidth=2)
            ax.set_title(f"{cfg_name.replace('_', ' ').title()} ({data_src})", fontsize=11, fontweight="bold")
            ax.set_xlabel("k (clusters)")
            ax.set_ylabel("Silhouette")
            ax.set_facecolor(colors["bg"])
            ax.grid(True, alpha=0.3)

            # Row 2: n_clusters
            ax = axes[1, col]
            ax.plot(ks, cluster_vals, "s-", color=colors["accent"], linewidth=2.5, markersize=8)
            ax.axvline(x=best_k, color=colors["primary"], linestyle="--", linewidth=2)
            ax.set_xlabel("k (clusters)")
            ax.set_ylabel("n_clusters")
            ax.set_facecolor(colors["bg"])
            ax.grid(True, alpha=0.3)

        fig.suptitle("K-Means Hyperparameter Tuning (Silhouette + n_clusters) — PhilGEPS (A–G)", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0.03, 0.06, 0.97, 0.92])
        summary_path = os.path.join(out_dir, "kmeans_hyperparam_tuning_summary.png")
        plt.savefig(summary_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        plt.close()
        print(f"K-Means: Saved combined hyperparameter tuning summary -> {summary_path}")


if __name__ == "__main__":
    main()
