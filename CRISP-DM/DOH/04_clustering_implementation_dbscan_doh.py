"""
Step 04b — DOH DBSCAN.

Density clustering on the same MinMax matrices as k-means. ε comes from k-NN distances with
multipliers; sweeps adapt when dimensionality or sample size would otherwise yield all-noise fits.
Noise points stay labeled -1.

Writes clustering_DOH_*_dbscan.csv and saves sensitivity / PCA panels under
webp/EDA_and_visualization/DOH/04_clustering_implementation/dbscan/.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# Shared `_common`, `sources_paths`, etc. live in parent `CRISP-DM/`.
script_dir = os.path.dirname(os.path.abspath(__file__))
_crisp_dm_root = os.path.dirname(script_dir)
if _crisp_dm_root not in sys.path:
    sys.path.insert(0, _crisp_dm_root)

from _common import _ensure_dir, _get_data_source_colors, data_source_for_dataset_name, get_doh_dataset_configs, load_features
from log_tee import tee_stdio_to_file
from sources_paths import data_root_doh, logs_dir_doh, webp_root_doh


def _suggest_eps(X: np.ndarray, k: int = 5) -> float:
    n_samples = X.shape[0]
    k = min(k, n_samples - 1)
    if k < 1:
        return 0.5
    n = min(n_samples, 1000)
    rng = np.random.default_rng(42)
    idx = rng.choice(n_samples, n, replace=False) if n_samples > n else np.arange(n_samples)
    X_sample = X[idx]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_sample)
    dists, _ = nn.kneighbors(X_sample)
    return float(np.median(dists[:, k]))


def fit_dbscan(X: np.ndarray, eps: float, min_samples: int):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(X)
    return db, labels


def _eps_isclose(a: float, b: float, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    return bool(np.isclose(a, b, rtol=rtol, atol=atol))


def _pairwise_distance_percentiles(
    X: np.ndarray,
    *,
    max_samples: int = 2000,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """p10 and p50 of pairwise distances on a row subsample (MinMax [0,1]^d geometry).

    k-distance eps can be orders of magnitude below typical inter-point distances when
    many rows share sparse/near-duplicate patterns; DBSCAN then leaves almost all points
    as noise unless eps_max reflects the wider dataset scale.
    """
    n = X.shape[0]
    m = min(max(50, n), max_samples)
    rng = np.random.default_rng(rng_seed)
    if n <= m:
        idx = np.arange(n)
    else:
        idx = rng.choice(n, m, replace=False)
    Y = X[idx]
    # Compute pairwise distances on a subsample (keeps this step bounded for large MinMax matrices).
    d2 = np.linalg.norm(Y[:, np.newaxis, :] - Y[np.newaxis, :, :], axis=2)
    triu = np.triu_indices(m, k=1)
    pdist = d2[triu]
    if pdist.size == 0:
        return 0.01, 0.5
    return float(np.percentile(pdist, 10)), float(np.percentile(pdist, 50))


def _run_eps_sensitivity(
    X: np.ndarray,
    min_samples: int,
    eps_suggested: float,
    n_eps: int = 25,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Sweep eps values and compute metrics.

    Returns:
      1) sensitivity_df with columns: eps, noise_ratio, assigned_ratio, n_clusters, silhouette
      2) all_labels: DBSCAN label vector for each eps row (same order as sensitivity_df)
    """
    d_p10, d_p50 = _pairwise_distance_percentiles(X)
    # Expand the search window: k-distance median often reflects local clumps only; merge
    # A/B-style data needs an upper bound in the ballpark of global pairwise scales.
    eps_min = max(1e-6, eps_suggested * 0.1)
    # eps_max uses both suggested k-distance scale and dataset-wide distance percentiles.
    eps_max = max(eps_suggested * 10.0, d_p10, d_p50 * 0.25)
    n_features = X.shape[1]
    eps_max = min(eps_max, float(np.sqrt(max(n_features, 1))) + 0.25)
    eps_values = np.linspace(eps_min, eps_max, n_eps)
    rows: list[dict[str, float | int]] = []
    all_labels: list[np.ndarray] = []
    n_samples = X.shape[0]

    for eps in eps_values:
        if eps <= 0:
            continue
        _, labels = fit_dbscan(X, float(eps), min_samples)
        all_labels.append(labels.copy())
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = int((labels == -1).sum())
        noise_ratio = noise_count / n_samples
        assigned_ratio = 1.0 - noise_ratio
        sil = 0.0
        # Silhouette is computed only on non-noise points; noise (-1) is excluded from the metric.
        if n_clusters >= 2 and (labels >= 0).sum() > 1:
            X_c = X[labels >= 0]
            labels_c = labels[labels >= 0]
            n_c = X_c.shape[0]
            if n_c > 5000:
                rng = np.random.default_rng(42)
                idx = rng.choice(n_c, min(5000, n_c), replace=False)
                sil = float(silhouette_score(X_c[idx], labels_c[idx]))
            else:
                sil = float(silhouette_score(X_c, labels_c))

        clustered_labels = labels[labels >= 0]
        largest_cluster_ratio = 0.0
        if clustered_labels.size > 0:
            _, counts = np.unique(clustered_labels, return_counts=True)
            if counts.size > 0:
                largest_cluster_ratio = float(counts.max() / n_samples)

        rows.append(
            {
                "eps": float(eps),
                "noise_ratio": float(noise_ratio),
                "assigned_ratio": float(assigned_ratio),
                "n_clusters": int(n_clusters),
                "silhouette": float(sil),
                "largest_cluster_ratio": float(largest_cluster_ratio),
            }
        )

    return pd.DataFrame(rows), all_labels


def _build_labels_for_plot_grid(
    sensitivity_df: pd.DataFrame,
    all_labels: list[np.ndarray],
    best_eps: float,
    best_labels: np.ndarray,
    plot_eps_count: int,
) -> list[tuple[float, np.ndarray]]:
    """Pick evenly spaced eps plus the sweep row closest to selected eps (counts as 'best' panel)."""
    n = len(all_labels)
    if n == 0 or sensitivity_df.empty:
        return []

    eps_arr = sensitivity_df["eps"].to_numpy(dtype=float)
    i_best = int(np.argmin(np.abs(eps_arr - float(best_eps))))

    if n <= plot_eps_count:
        plot_indices = list(range(n))
    else:
        plot_indices = np.linspace(0, n - 1, plot_eps_count, dtype=int).tolist()
    plot_indices.append(i_best)
    plot_indices = sorted({int(i) for i in plot_indices if 0 <= int(i) < n})

    out: list[tuple[float, np.ndarray]] = []
    for i in plot_indices:
        ep = float(sensitivity_df.iloc[i]["eps"])
        lab = best_labels if i == i_best else all_labels[i]
        out.append((ep, lab))
    return out


def _build_labels_for_plot_full_resample(
    X: np.ndarray,
    sensitivity_df: pd.DataFrame,
    best_eps: float,
    best_labels_full: np.ndarray,
    min_samples: int,
    plot_eps_count: int,
) -> list[tuple[float, np.ndarray]]:
    """Re-fit DBSCAN on full X for each panel (used when eps sweep ran on a row subsample)."""
    n = len(sensitivity_df)
    if n == 0 or sensitivity_df.empty:
        return [(float(best_eps), best_labels_full)]

    eps_arr = sensitivity_df["eps"].to_numpy(dtype=float)
    i_best = int(np.argmin(np.abs(eps_arr - float(best_eps))))

    if n <= plot_eps_count:
        plot_indices = list(range(n))
    else:
        plot_indices = np.linspace(0, n - 1, plot_eps_count, dtype=int).tolist()
    plot_indices.append(i_best)
    plot_indices = sorted({int(i) for i in plot_indices if 0 <= int(i) < n})

    out: list[tuple[float, np.ndarray]] = []
    for i in plot_indices:
        ep = float(sensitivity_df.iloc[i]["eps"])
        if i == i_best:
            out.append((ep, best_labels_full))
        else:
            _, lab = fit_dbscan(X, ep, min_samples)
            out.append((ep, lab))
    return out


def _plot_eps_sensitivity(
    sensitivity_df: pd.DataFrame,
    cfg_name: str,
    best_eps: float,
    out_path: str,
    data_source: str,
) -> None:
    """Plot eps vs Noise/Assigned ratio and n_clusters for DBSCAN hyperparameter tuning."""
    colors = _get_data_source_colors(data_source)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), dpi=160, facecolor=colors["bg"], sharex=True)
    eps_vals = sensitivity_df["eps"].values

    # Noise ratio
    ax = axes[0]
    ax.plot(eps_vals, sensitivity_df["noise_ratio"], "o-", color="#e74c3c", linewidth=2, markersize=4)
    ax.set_ylabel("Noise ratio\n(lower = better)", fontsize=10, color="#e74c3c")
    ax.tick_params(axis="y", labelcolor="#e74c3c")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.4, color="#e74c3c", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=best_eps, color=colors["primary"], linestyle=":", linewidth=2)
    ax.grid(True, alpha=0.3)

    # Assigned ratio
    ax = axes[1]
    ax.plot(eps_vals, sensitivity_df["assigned_ratio"], "s-", color="#27ae60", linewidth=2, markersize=4)
    ax.set_ylabel("Assigned ratio\n(higher = better)", fontsize=10, color="#27ae60")
    ax.tick_params(axis="y", labelcolor="#27ae60")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.6, color="#27ae60", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=best_eps, color=colors["primary"], linestyle=":", linewidth=2)
    ax.grid(True, alpha=0.3)

    # Number of clusters
    ax = axes[2]
    ax.plot(eps_vals, sensitivity_df["n_clusters"], "^-", color=colors["accent"], linewidth=2, markersize=4)
    ax.set_ylabel("n_clusters\n(excl. noise)", fontsize=10)
    ax.set_xlabel("epsilon (eps)", fontsize=11)
    ax.axvline(x=best_eps, color=colors["primary"], linestyle=":", linewidth=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"DBSCAN Hyperparameter Tuning - {cfg_name}\nSelected eps={best_eps:.4f} | Source: {data_source}",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )

    for ax in axes:
        ax.set_facecolor(colors["bg"])

    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def _plot_dbscan_clustering_results_across_eps(
    *,
    X_2d: np.ndarray,
    labels_for_plot: list[tuple[float, np.ndarray]],
    best_eps: float,
    cfg_name: str,
    out_path: str,
    data_source: str,
) -> None:
    """Grid of PCA(2D) clustering results across varying eps."""
    if not labels_for_plot:
        return

    _ensure_dir(os.path.dirname(out_path))
    colors = _get_data_source_colors(data_source)

    labels_for_plot_sorted = sorted(labels_for_plot, key=lambda x: x[0])
    n = len(labels_for_plot_sorted)
    n_cols = min(5, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.2 * n_rows),
        dpi=160,
        facecolor=colors["bg"],
        sharex=True,
        sharey=True,
    )
    if n_rows == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, (eps, labels) in enumerate(labels_for_plot_sorted):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]

        mask_noise = labels == -1
        mask_cluster = ~mask_noise

        if mask_noise.any():
            ax.scatter(
                X_2d[mask_noise, 0],
                X_2d[mask_noise, 1],
                c="#95a5a6",
                alpha=0.35,
                s=12,
                edgecolors="none",
            )

        if mask_cluster.any():
            ax.scatter(
                X_2d[mask_cluster, 0],
                X_2d[mask_cluster, 1],
                c=labels[mask_cluster],
                cmap="tab10",
                alpha=0.75,
                s=18,
                edgecolors="white",
                linewidth=0.25,
            )

        is_best = _eps_isclose(eps, best_eps)
        ax.set_title(f"eps={eps:.4f}" + (" (best)" if is_best else ""), fontsize=10, fontweight="bold")
        ax.set_facecolor(colors["bg"])
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        if is_best:
            for spine in ax.spines.values():
                spine.set_linewidth(2.2)
                spine.set_color(colors["primary"])

    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"DBSCAN clustering results across eps (PCA 2D) - {cfg_name}\nSelected eps={best_eps:.4f} | Source: {data_source}",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def _select_best_eps(sensitivity_df: pd.DataFrame, eps_suggested: float) -> float:
    """Choose eps using silhouette penalized by noise.

    DBSCAN silhouette is computed on clustered points only (noise excluded), so high
    silhouette can still occur even when most points are noise. To avoid the
    under-clustering failure mode, we weight silhouette by `assigned_ratio`.
    """
    candidates = sensitivity_df[sensitivity_df["n_clusters"] >= 2].copy()
    if candidates.empty:
        candidates = sensitivity_df.copy()

    # Weighted score:
    # - silhouette (computed excl. noise) measures separation of clustered points
    # - assigned_ratio penalizes configurations where most points are noise
    # - largest_cluster_ratio penalizes fragmentation into many tiny clusters
    if "largest_cluster_ratio" in candidates.columns:
        # Penalize solutions that fragment into many tiny clusters.
        candidates["score"] = candidates["silhouette"] * candidates["assigned_ratio"] * np.sqrt(candidates["largest_cluster_ratio"])
    else:
        candidates["score"] = candidates["silhouette"] * candidates["assigned_ratio"]
    best_row = candidates.loc[candidates["score"].idxmax()]
    if pd.isna(best_row.get("eps")):
        return eps_suggested
    return float(best_row["eps"])


def main() -> None:
    # --- Step 0: Setup paths ---
    cluster_dir = os.path.join(data_root_doh(), "04_clustering")
    out_dir = os.path.join(webp_root_doh(), "04_clustering_implementation", "dbscan")
    _ensure_dir(cluster_dir)
    _ensure_dir(out_dir)
    term_log = os.path.join(logs_dir_doh(), "04_clustering_implementation_dbscan_doh_terminal.txt")
    with tee_stdio_to_file(term_log):
        _run_dbscan_doh(cluster_dir, out_dir)


def _run_dbscan_doh(cluster_dir: str, out_dir: str) -> None:
    for cfg in get_doh_dataset_configs():
        # --- Step 1: Load minmax-scaled features ---
        result = load_features(cfg)
        if result is None:
            print(f"DBSCAN [{cfg.name}]: Skip (file missing or empty)")
            continue

        df, _, X = result
        n_samples, n_features = X.shape
        data_src = data_source_for_dataset_name(cfg.name)

        # --- Step 2: min_samples and k-distance for initial eps ---
        # High-D PhilGEPS: 10+ is often too strict with moderate eps; low-D DOH: need >=4 to avoid over-merge.
        min_samples = max(3, min(8, max(4, (n_features + 5) // 4)))
        k = min(5, n_samples - 1)
        eps_suggested = _suggest_eps(X, k=k)

        # --- Step 3: Eps sensitivity (subsample very large n; final fit uses full X) ---
        sweep_cap = 12000
        use_sweep_subsample = n_samples > sweep_cap
        if use_sweep_subsample:
            rng = np.random.default_rng(42)
            sub_ix = rng.choice(n_samples, sweep_cap, replace=False)
            X_sweep = X[sub_ix]
            n_eps = 12
            print(f"DBSCAN [{cfg.name}]: eps sweep on subsample n={sweep_cap}/{n_samples} (full-data fit after selection)", flush=True)
        else:
            X_sweep = X
            n_eps = 15 if n_samples > 10000 else 25

        sensitivity_df, all_labels_sub = _run_eps_sensitivity(
            X_sweep,
            min_samples,
            eps_suggested,
            n_eps=n_eps,
        )
        sensitivity_csv = os.path.join(out_dir, f"{cfg.name}_eps_sensitivity.csv")
        sensitivity_df.to_csv(sensitivity_csv, index=False)
        print(f"DBSCAN [{cfg.name}]: Saved eps sensitivity -> {sensitivity_csv}", flush=True)

        # --- Step 4: Select best eps (noise/assigned constraints + silhouette) ---
        # `best_eps` is chosen to balance cluster separation (silhouette), coverage (assigned_ratio),
        # and stability (avoid extreme fragmentation via largest_cluster_ratio).
        best_eps = _select_best_eps(sensitivity_df, eps_suggested)

        # --- Step 5: Fit DBSCAN with selected eps ---
        print(f"DBSCAN [{cfg.name}]: fitting full data with selected eps={best_eps:.6f} ...", flush=True)
        _, best_labels = fit_dbscan(X, best_eps, min_samples)
        best_n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
        best_noise_ratio = (best_labels == -1).sum() / n_samples

        # --- Step 6: Visualize DBSCAN clustering results across varying eps (PCA 2D grid) ---
        if use_sweep_subsample:
            if n_samples > 20000:
                labels_for_plot = [(float(best_eps), best_labels)]
            else:
                labels_for_plot = _build_labels_for_plot_full_resample(
                    X, sensitivity_df, best_eps, best_labels, min_samples, plot_eps_count=4
                )
        else:
            labels_for_plot = _build_labels_for_plot_grid(
                sensitivity_df, all_labels_sub, best_eps, best_labels, plot_eps_count=5
            )

        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
        eps_results_path = os.path.join(out_dir, f"{cfg.name}_clustering_results_across_eps.png")
        _plot_dbscan_clustering_results_across_eps(
            X_2d=X_2d,
            labels_for_plot=labels_for_plot,
            best_eps=best_eps,
            cfg_name=cfg.name,
            out_path=eps_results_path,
            data_source=data_src,
        )
        print(f"DBSCAN [{cfg.name}]: Saved clustering results across eps -> {eps_results_path}")

        # --- Step 6: Eps vs Noise/Assigned Ratio transition plot (thesis-ready) ---
        eps_sens_path = os.path.join(out_dir, f"{cfg.name}_eps_sensitivity_plot.png")
        _plot_eps_sensitivity(sensitivity_df, cfg.name, best_eps, eps_sens_path, data_src)
        print(f"DBSCAN [{cfg.name}]: Saved eps sensitivity plot -> {eps_sens_path}")

        # --- Step 7: Save cluster labels and params ---
        # Persist the chosen clustering assignment back onto the original feature rows.
        df_out = df.copy()
        df_out["cluster_dbscan"] = best_labels
        out_csv = os.path.join(cluster_dir, f"clustering_{cfg.name}_dbscan.csv")
        df_out.to_csv(out_csv, index=False)

        noise_count = int((best_labels == -1).sum())
        sil = 0.0
        if best_n_clusters >= 2:
            mask = best_labels >= 0
            Xc, lc = X[mask], best_labels[mask]
            nc = Xc.shape[0]
            if nc > 5000:
                rng = np.random.default_rng(43)
                idx = rng.choice(nc, 5000, replace=False)
                sil = float(silhouette_score(Xc[idx], lc[idx]))
            elif nc > 1:
                sil = float(silhouette_score(Xc, lc))

        print(
            f"DBSCAN [{cfg.name}]: eps={best_eps:.4f}, min_samples={min_samples}, clusters={best_n_clusters}, noise={noise_count} ({best_noise_ratio*100:.1f}%), sil={sil:.4f} -> {out_csv}",
            flush=True,
        )
        with open(os.path.join(out_dir, f"{cfg.name}_params.txt"), "w", encoding="utf-8") as f:
            f.write(
                f"eps={best_eps}\nmin_samples={min_samples}\nn_clusters={best_n_clusters}\nnoise_count={noise_count}\nnoise_ratio={best_noise_ratio:.4f}\nsilhouette={sil:.4f}\nselection_method=silhouette_x_assigned_ratio\noutput={out_csv}\n"
            )
            if use_sweep_subsample:
                f.write(f"\neps_sweep_rows_subsampled={sweep_cap}\n")
            f.write(f"\n# Parameter selection: k-distance median -> eps sensitivity (noise/assigned ratio) -> best eps\n")


if __name__ == "__main__":
    main()
