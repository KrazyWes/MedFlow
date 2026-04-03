"""
Step 8 — package clustering results for writing (thesis / defense slides).

For each `DatasetConfig` and each algorithm (`kmeans`, `dbscan`), reads the labeled CSV from
`this_datasets/.../04_clustering/`, builds plots under `webp/.../08_output_to_use/{algo}/{slug}/`,
and writes `export_labeled.csv` (IDs + numeric features + cluster column).

Pieces:
  - Thematic scores (`_thematic_top10_*`) align the top-10 bar and the “flagged” PCA overlay
    with the story for that lens (shortage, overstock, …); if no rule matches, fall back to L2 norm.
  - PCA scatter is fit on the same feature matrix used for clustering (already scaled upstream).
  - Silhouette plot subsamples very large n so matplotlib stays responsive.

Entry point: `run_bundles_for_source("DOH" | "PhilGEPS")` from `08_final_output_bundle_*.py`.
"""

from __future__ import annotations

import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from _common import (
    DatasetConfig,
    _add_data_source_badge,
    _ensure_dir,
    _get_data_source_colors,
    FIG_RECT_HEATMAP,
    FIG_RECT_LEGEND_RIGHT,
    chart_bar_edge_color,
    chart_distinct_colors,
    chart_gradient_bar_colors,
    data_source_for_dataset_name,
)

Source = Literal["DOH", "PhilGEPS"]

LABEL_COL = {"kmeans": "cluster_kmeans", "dbscan": "cluster_dbscan"}

TOP_N = 10


# --- matplotlib colormap shim (older vs newer mpl APIs) ---
def _get_cmap(name: str):
    cm = getattr(plt, "colormaps", None)
    if cm is not None and hasattr(cm, "get_cmap"):
        try:
            return cm.get_cmap(name)
        except Exception:
            pass
    return plt.cm.get_cmap(name)


RADAR_MAX_FEATURES = 12
SILHOUETTE_PLOT_MAX_SAMPLES = 2500


# --- Naming / column helpers ---
def _bundle_slug(cfg_name: str) -> str:
    for prefix in ("DOH_", "PhilGEPS_"):
        if cfg_name.startswith(prefix):
            return cfg_name[len(prefix) :]
    return cfg_name


def _col_num(df: pd.DataFrame, col: str) -> np.ndarray | None:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _thematic_top10_scores(df: pd.DataFrame, slug: str) -> tuple[np.ndarray, str] | None:
    """
    Per-slug scoring for who belongs in the top-10 bar chart.
    Mirrors how step 02 engineered features for that lens. Returns None -> caller uses L2 norm of rows in X.
    """
    s = slug.lower()
    if "high_risk_shortage" in s:
        for c in ("feature_shortage_depth", "feature_supply_shortage_proxy"):
            v = _col_num(df, c)
            if v is not None:
                # Depth is negative for low supply; negate so higher score = stronger shortage signal
                return (-v, f"Thematic score: −({c}) — higher = higher shortage risk")
    if "overstocking" in s:
        for c in ("feature_overstock_intensity", "feature_procurement_intensity"):
            v = _col_num(df, c)
            if v is not None:
                return (v, f"Thematic score: {c} — higher = stronger overstock / intensity")
    if "inefficient_distribution" in s:
        for c in ("feature_deliveries_per_unit_qty", "feature_lines_per_unit_qty"):
            v = _col_num(df, c)
            if v is not None:
                return (v, f"Thematic score: {c} — higher = more deliveries per unit quantity")
    if "unequal_supply" in s:
        conc = [
            c
            for c in df.columns
            if isinstance(c, str)
            and c.startswith("feature_")
            and ("per_unique" in c or "per_award" in c)
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if conc:
            sub = df[conc].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            return (sub.abs().sum(axis=1).to_numpy(), "Thematic score: Σ|concentration features| — higher = more uneven / concentrated pattern")
    if "distribution_recipient" in s:
        q = _col_num(df, "quantity_total")
        a = _col_num(df, "total_amount_total")
        if a is None:
            a = _col_num(df, "contract_amount_total")
        if q is not None and a is not None:
            return (
                np.log1p(np.maximum(q, 0)) + np.log1p(np.maximum(a, 0)),
                "Thematic score: log1p(quantity) + log1p(amount) — higher = larger distribution / contract volume",
            )
        if q is not None:
            return (np.log1p(np.maximum(q, 0)), "Thematic score: log1p(quantity_total)")
    if "supplier_awardee" in s:
        v = _col_num(df, "contract_amount_total")
        if v is not None:
            return (np.log1p(np.maximum(v, 0)), "Thematic score: log1p(contract_amount_total) — higher = larger total contract value")
    if "medicine_procurement" in s:
        ib = _col_num(df, "item_budget")
        qn = _col_num(df, "quantity")
        score_ib = np.log1p(np.maximum(ib, 0)) if ib is not None else None
        score_q = np.log1p(np.maximum(qn, 0)) if qn is not None else None
        if score_ib is not None and score_q is not None:
            return (score_ib + score_q, "Thematic score: log1p(item_budget) + log1p(quantity)")
        if score_ib is not None:
            return (score_ib, "Thematic score: log1p(item_budget)")
    return None


def _flagging_scores(df: pd.DataFrame, slug: str, X: np.ndarray) -> tuple[np.ndarray, str]:
    """Scores for highlighting the same extreme units as top10_bar; returns (scores, short legend note)."""
    t = _thematic_top10_scores(df, slug)
    if t is not None:
        scores, xlabel = t
        note = xlabel.split("—")[0].strip() if "—" in xlabel else "thematic score"
        if len(note) > 52:
            note = note[:49] + "…"
        return np.asarray(scores, dtype=float), note
    return np.linalg.norm(X, axis=1), "L2 norm (extremity)"


def _id_label_row(row: pd.Series, id_columns: list[str]) -> str:
    parts = []
    for c in id_columns:
        if c in row.index:
            v = row[c]
            if pd.isna(v):
                parts.append("")
            else:
                s = str(v).strip()
                parts.append(s[:60] if len(s) > 60 else s)
    return " | ".join(p for p in parts if p) or "(no id)"


def _numeric_feature_cols(df: pd.DataFrame, id_columns: list[str], label_col: str) -> list[str]:
    skip = set(id_columns) | {label_col, "cluster_kmeans", "cluster_dbscan"}
    out: list[str] = []
    for c in df.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def _mask_valid_clusters(labels: np.ndarray) -> np.ndarray:
    """DBSCAN noise is -1; silhouette and centroids usually ignore those rows."""
    return labels >= 0


def _subsample_stratified(X: np.ndarray, labels: np.ndarray, max_n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[0] <= max_n:
        return X, labels
    uniq = np.unique(labels)
    weights = np.array([(labels == k).sum() for k in uniq], dtype=float)
    weights = weights / max(weights.sum(), 1.0)
    counts = np.maximum((weights * max_n).astype(int), 1)
    fix = counts.sum() - max_n
    while fix > 0:
        i = int(np.argmax(counts))
        if counts[i] > 1:
            counts[i] -= 1
            fix -= 1
        else:
            break
    idx_list: list[int] = []
    for k, n_take in zip(uniq, counts):
        inds = np.where(labels == k)[0]
        if len(inds) <= n_take:
            idx_list.extend(inds.tolist())
        else:
            pick = rng.choice(inds, n_take, replace=False)
            idx_list.extend(pick.tolist())
    idx_arr = np.array(sorted(idx_list), dtype=int)
    return X[idx_arr], labels[idx_arr]


# --- Figures (PCA, bars, silhouette, heatmap, radar) ---
def _plot_scatter_pca2d(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    out_path: str,
    title: str,
    colors: dict,
    data_src: str,
) -> None:
    if X.shape[0] < 2 or X.shape[1] < 1:
        return
    n_comp = min(2, X.shape[1], max(1, X.shape[0] - 1))
    if n_comp < 2:
        xcoord = X[:, 0]
        ycoord = np.zeros_like(xcoord)
    else:
        pca = PCA(n_components=2, random_state=42)
        xy = pca.fit_transform(X)
        xcoord, ycoord = xy[:, 0], xy[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=160, facecolor=colors["bg"])
    labs = labels.astype(object)
    uniq = sorted(np.unique(labs), key=lambda v: (str(v) == "-1", str(v)))
    for i, lab in enumerate(uniq):
        m = labs == lab
        ax.scatter(
            xcoord[m],
            ycoord[m],
            s=22,
            alpha=0.72,
            label=str(lab),
            color=f"C{i % 10}",
            edgecolors="white",
            linewidths=0.25,
        )
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_facecolor(colors["bg"])
    ax.legend(
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=8,
        ncol=1,
        frameon=True,
    )
    plt.tight_layout(rect=FIG_RECT_LEGEND_RIGHT)
    _add_data_source_badge(ax, data_src)
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def _plot_scatter_pca2d_flagged(
    X: np.ndarray,
    labels: np.ndarray,
    flagged_mask: np.ndarray,
    *,
    out_path: str,
    title: str,
    colors: dict,
    data_src: str,
    flag_caption: str,
) -> None:
    """PCA 2D cluster scatter with the flagged subset drawn as high-visibility red-ring markers."""
    if X.shape[0] < 2 or X.shape[1] < 1:
        return
    n_comp = min(2, X.shape[1], max(1, X.shape[0] - 1))
    if n_comp < 2:
        xcoord = X[:, 0]
        ycoord = np.zeros_like(xcoord)
    else:
        pca = PCA(n_components=2, random_state=42)
        xy = pca.fit_transform(X)
        xcoord, ycoord = xy[:, 0], xy[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=160, facecolor=colors["bg"])
    labs = labels.astype(object)
    uniq = sorted(np.unique(labs), key=lambda v: (str(v) == "-1", str(v)))
    for i, lab in enumerate(uniq):
        m = labs == lab
        ax.scatter(
            xcoord[m],
            ycoord[m],
            s=20,
            alpha=0.5,
            label=str(lab),
            color=f"C{i % 10}",
            edgecolors="white",
            linewidths=0.2,
        )

    fm = np.asarray(flagged_mask, dtype=bool)
    if fm.any():
        ax.scatter(
            xcoord[fm],
            ycoord[fm],
            s=120,
            facecolors="none",
            edgecolors="#c0392b",
            linewidths=2.0,
            alpha=0.95,
            label="Flagged set",
            zorder=5,
        )

    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_facecolor(colors["bg"])
    ax.legend(
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=8,
        ncol=1,
        frameon=True,
    )
    cap_line = flag_caption if len(flag_caption) <= 130 else flag_caption[:127] + "…"
    plt.tight_layout(rect=[0.05, 0.24, 0.72, 0.94])
    _add_data_source_badge(ax, data_src, footer_y=0.12)
    fig = ax.get_figure()
    fig.text(0.5, 0.045, cap_line, transform=fig.transFigure, ha="center", va="bottom", fontsize=7, color="#333333")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def _plot_top10(
    df: pd.DataFrame,
    X: np.ndarray,
    thematic_slug: str,
    *,
    id_columns: list[str],
    out_path: str,
    title: str,
    colors: dict,
) -> None:
    thematic = _thematic_top10_scores(df, thematic_slug)
    if thematic is not None:
        scores_all, xlabel = thematic
        scores_all = np.asarray(scores_all, dtype=float)
    else:
        scores_all = np.linalg.norm(X, axis=1)
        xlabel = "L2 norm of scaled feature vector (fallback — higher = more extreme profile)"

    order = np.argsort(-scores_all)
    top_ix = order[:TOP_N]
    sub = df.iloc[top_ix].reset_index(drop=True)
    scores = scores_all[top_ix]
    labels_y = [_id_label_row(sub.iloc[i], id_columns) for i in range(len(sub))]
    labels_y = [s if len(s) <= 42 else s[:39] + "…" for s in labels_y]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160, facecolor=colors["bg"])
    y = np.arange(len(labels_y))
    # Strongest score = deeper primary tint; lowest of the ten = closer to secondary (rank gradient).
    bar_cols = chart_gradient_bar_colors(colors, len(y), flip=False)
    bar_edges = [chart_bar_edge_color(c) for c in bar_cols]
    ax.barh(y, scores, color=bar_cols, edgecolor=bar_edges, linewidth=1.05)
    ax.set_yticks(y)
    ax.set_yticklabels(labels_y, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(f"{title} (Top {TOP_N})", fontsize=12, fontweight="bold")
    ax.set_facecolor(colors["bg"])
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def _silhouette_placeholder(out_path: str, title: str, note: str, colors: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=150, facecolor=colors["bg"])
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight="bold", color=colors.get("primary", "#1a1a1a"))
    ax.text(0.5, 0.5, note, ha="center", va="center", fontsize=11, color="#333333", transform=ax.transAxes, wrap=True)
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def _plot_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    out_path: str,
    title: str,
    colors: dict,
) -> None:
    mask = labels >= 0
    Xv = X[mask]
    labv = labels[mask]
    if Xv.shape[0] < 3 or len(np.unique(labv)) < 2:
        _silhouette_placeholder(
            out_path,
            title,
            "Silhouette diagram not produced: need at least 3 non-noise points and 2 distinct clusters.",
            colors,
        )
        return

    rng = np.random.default_rng(42)
    Xp, lp = _subsample_stratified(Xv, labv, SILHOUETTE_PLOT_MAX_SAMPLES, rng)

    s_vals = silhouette_samples(Xp, lp, metric="euclidean")
    order = np.argsort(lp, kind="stable")
    s_vals = s_vals[order]
    lp = lp[order]

    uniq = np.unique(lp)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160, facecolor=colors["bg"])
    y_lower = 10
    viridis = _get_cmap("viridis")
    for i, cid in enumerate(uniq):
        m = lp == cid
        vals = np.sort(s_vals[m])
        size = vals.size
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.82, color=viridis(i / max(len(uniq) - 1, 1)))
        ax.text(-0.06, y_lower + 0.5 * size, str(cid), fontsize=9, va="center")
        y_lower = y_upper + 10
    ax.axvline(float(np.mean(s_vals)), color="red", linestyle="--", linewidth=1.6, label="mean silhouette")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Silhouette coefficient", fontsize=11)
    ax.set_ylabel("Samples (grouped by cluster)", fontsize=10)
    ax.set_facecolor(colors["bg"])
    ax.legend(bbox_to_anchor=(1.02, 0.45), loc="center left", fontsize=9, frameon=True)
    plt.tight_layout(rect=[0.12, 0.08, 0.8, 0.93])
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def _plot_centroid_heatmap(
    centroids: np.ndarray,
    feature_names: list[str],
    cluster_ids: list,
    *,
    out_path: str,
    title: str,
    colors: dict,
) -> None:
    if centroids.size == 0:
        return
    # Optionally truncate display names
    disp_names = [n[:24] + "…" if len(n) > 24 else n for n in feature_names]
    fig, ax = plt.subplots(figsize=(min(20, 6 + 0.22 * len(disp_names)), max(4, 0.55 * len(cluster_ids))), dpi=160, facecolor=colors["bg"])
    im = ax.imshow(centroids, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels([str(c) for c in cluster_ids], fontsize=10)
    ax.set_xticks(range(len(disp_names)))
    ax.set_xticklabels(disp_names, rotation=75, ha="right", fontsize=7)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    ax.set_facecolor(colors["bg"])
    plt.tight_layout(rect=FIG_RECT_HEATMAP)
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def _plot_radar(
    centroids: np.ndarray,
    feature_names: list[str],
    cluster_ids: list,
    *,
    out_path: str,
    title: str,
    colors: dict,
) -> None:
    if centroids.shape[0] == 0 or centroids.shape[1] == 0:
        return
    n_feat = centroids.shape[1]
    if n_feat > RADAR_MAX_FEATURES:
        var = np.var(centroids, axis=0)
        pick = np.argsort(-var)[:RADAR_MAX_FEATURES]
        pick.sort()
        centroids = centroids[:, pick]
        feature_names = [feature_names[i] for i in pick]
    # min-max scale each column across clusters for shape comparison
    col_min = centroids.min(axis=0)
    col_max = centroids.max(axis=0)
    denom = np.where(col_max - col_min < 1e-12, 1.0, col_max - col_min)
    Z = (centroids - col_min) / denom

    n_axes = Z.shape[1]
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.2, 7.2), dpi=160, subplot_kw=dict(projection="polar"), facecolor=colors["bg"])
    line_cols = chart_distinct_colors(len(Z))
    for i, row in enumerate(Z):
        vals = row.tolist()
        vals += vals[:1]
        lc = line_cols[i]
        ec = chart_bar_edge_color(lc, factor=0.3)
        ax.plot(
            angles,
            vals,
            "o-",
            linewidth=2.8,
            label=f"C {cluster_ids[i]}",
            color=lc,
            markerfacecolor="white",
            markeredgecolor=ec,
            markeredgewidth=1.35,
            markersize=5,
        )
        ax.fill(angles, vals, alpha=0.14, color=lc)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(np.linspace(0, 2 * np.pi, n_axes, endpoint=False)), feature_names, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(title + "\n(per-feature min-max across clusters)", fontsize=11, fontweight="bold", pad=16)
    ax.legend(loc="upper left", bbox_to_anchor=(1.32, 1.06), fontsize=8, frameon=True)
    ax.set_facecolor(colors["bg"])
    plt.tight_layout(rect=[0.04, 0.06, 0.66, 0.9])
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def _export_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def process_dataset_bundle(
    cfg: DatasetConfig,
    cluster_dir: str,
    out_root: str,
    algo: str,
) -> None:
    """Load one labeled clustering CSV and emit the full 08 artifact set (or skip with a reason)."""
    label_col = LABEL_COL[algo]
    cluster_path = os.path.join(cluster_dir, f"clustering_{cfg.name}_{algo}.csv")
    if not os.path.isfile(cluster_path):
        print(f"Bundle [{cfg.name}][{algo}]: skip (missing {cluster_path})")
        return

    df = pd.read_csv(cluster_path, low_memory=False)
    if label_col not in df.columns:
        print(f"Bundle [{cfg.name}][{algo}]: skip (no {label_col})")
        return

    id_cols = [c for c in cfg.id_columns if c in df.columns]
    feat_cols = _numeric_feature_cols(df, id_cols, label_col)
    if not feat_cols:
        print(f"Bundle [{cfg.name}][{algo}]: skip (no numeric features)")
        return

    labels = df[label_col].to_numpy()
    X = df[feat_cols].to_numpy(dtype=float)
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    data_src = data_source_for_dataset_name(cfg.name)
    colors = _get_data_source_colors(data_src)
    slug = _bundle_slug(cfg.name)
    out_dir = os.path.join(out_root, "08_output_to_use", algo, slug)
    _ensure_dir(out_dir)

    valid = _mask_valid_clusters(labels)
    n_clusters = len(np.unique(labels[valid])) if valid.any() else 0
    if algo == "kmeans" and n_clusters < 2:
        print(f"Bundle [{cfg.name}][{algo}]: skip (<2 clusters)")
        return
    if algo == "dbscan" and n_clusters < 1:
        print(f"Bundle [{cfg.name}][{algo}]: skip (no dense clusters)")
        return

    title_base = f"{algo.upper()} — {cfg.name.replace('_', ' ')}"

    # Order: quick global views first, then thematic top-10, silhouette, centroid-based charts, CSV.
    _plot_scatter_pca2d(
        X,
        labels,
        out_path=os.path.join(out_dir, "cluster_scatter_pca2d.png"),
        title=f"{title_base} (PCA 2D)",
        colors=colors,
        data_src=data_src,
    )

    scores_rank, score_note = _flagging_scores(df, slug, X)
    k_flag = min(TOP_N, int(scores_rank.shape[0]))
    flagged_mask = np.zeros(scores_rank.shape[0], dtype=bool)
    if k_flag > 0:
        top_ix = np.argsort(-scores_rank)[:k_flag]
        flagged_mask[top_ix] = True
    _plot_scatter_pca2d_flagged(
        X,
        labels,
        flagged_mask,
        out_path=os.path.join(out_dir, "cluster_scatter_pca2d_flagged.png"),
        title=f"{title_base} (PCA 2D — flagged set)",
        colors=colors,
        data_src=data_src,
        flag_caption=f"Flagged: top {k_flag} by {score_note} (same rule as top10_bar)",
    )

    _plot_top10(
        df,
        X,
        slug,
        id_columns=id_cols,
        out_path=os.path.join(out_dir, "top10_bar.png"),
        title=title_base,
        colors=colors,
    )

    _plot_silhouette(
        X,
        labels,
        out_path=os.path.join(out_dir, "silhouette_plot.png"),
        title=f"{title_base} — silhouette",
        colors=colors,
    )

    cl_ids = sorted({int(x) for x in np.unique(labels[valid]).tolist()})
    if cl_ids:
        centroids = np.array([X[(labels == c) & valid].mean(axis=0) for c in cl_ids])
        _plot_centroid_heatmap(
            centroids,
            feat_cols,
            cl_ids,
            out_path=os.path.join(out_dir, "clustermap_centroids.png"),
            title=f"{title_base} — centroid heatmap",
            colors=colors,
        )
        max_radar_clusters = 10
        if len(cl_ids) > max_radar_clusters:
            # Radar gets unreadable with dozens of series; keep the largest clusters by headcount.
            sizes = np.array([(labels == c).sum() for c in cl_ids])
            topc = np.argsort(-sizes)[:max_radar_clusters]
            cl_ids_r = [cl_ids[i] for i in sorted(topc.tolist())]
            centroids_r = np.array([X[labels == c].mean(axis=0) for c in cl_ids_r])
        else:
            cl_ids_r = cl_ids
            centroids_r = centroids
        _plot_radar(
            centroids_r,
            feat_cols,
            cl_ids_r,
            out_path=os.path.join(out_dir, "web_chart_radar.png"),
            title=f"{title_base} — radar (web chart)",
            colors=colors,
        )

    export_cols = [c for c in id_cols + feat_cols + [label_col] if c in df.columns]
    _export_csv(df[export_cols], os.path.join(out_dir, "export_labeled.csv"))

    print(f"Bundle [{cfg.name}][{algo}] -> {out_dir}")


def run_bundles_for_source(source: Source) -> None:
    """Loop all dataset configs for one source × both algorithms."""
    from sources_paths import data_root_doh, data_root_philgeps, webp_root_doh, webp_root_philgeps

    if source == "DOH":
        from _common import get_doh_dataset_configs

        cfgs = get_doh_dataset_configs()
        cluster_dir = os.path.join(data_root_doh(), "04_clustering")
        out_root = webp_root_doh()
    else:
        from _common import get_philgeps_dataset_configs

        cfgs = get_philgeps_dataset_configs()
        cluster_dir = os.path.join(data_root_philgeps(), "04_clustering")
        out_root = webp_root_philgeps()

    for cfg in cfgs:
        for algo in ("kmeans", "dbscan"):
            process_dataset_bundle(cfg, cluster_dir, out_root, algo)
    print(f"Step 8 output bundles finished for {source}.")
