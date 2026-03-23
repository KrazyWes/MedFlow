"""
Shared utilities for CRISP-DM pipeline (Data Cleaning, Transformation, EDA, Clustering).
"""

from __future__ import annotations

import atexit
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore[import-not-found]


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------------------------------------------------------
# Step: Data loaders - discover and load CSV/Excel files from raw datasets
# -----------------------------------------------------------------------------
CSV_EXTENSIONS = (".csv",)
EXCEL_EXTENSIONS = (".xlsx", ".xls")
ALL_DATA_EXTENSIONS = CSV_EXTENSIONS + EXCEL_EXTENSIONS


def get_supported_files(directory: str) -> list[tuple[str, str]]:
    """Return list of (filepath, extension) for CSV/Excel files in directory."""
    if not os.path.isdir(directory):
        return []
    files = []
    for f in os.listdir(directory):
        fp = os.path.join(directory, f)
        if os.path.isfile(fp):
            ext = os.path.splitext(f)[1].lower()
            if ext in ALL_DATA_EXTENSIONS:
                files.append((fp, ext))
    return sorted(files, key=lambda x: x[0])


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def load_excel(path: str, dtype=None) -> list[pd.DataFrame]:
    xl = pd.ExcelFile(path)
    kwargs = {"dtype": dtype} if dtype is not None else {}
    return [pd.read_excel(path, sheet_name=name, **kwargs) for name in xl.sheet_names]


# -----------------------------------------------------------------------------
# Step: EDA utilities - missingness, histograms, categorical bars, correlation
# -----------------------------------------------------------------------------
# Color palette and style: DOH (teal/blue), PhilGEPS (green/emerald)
EDA_COLORS = {
    "DOH": {"primary": "#1a5276", "secondary": "#3498db", "accent": "#5dade2", "bg": "#eaf2f8"},
    "PhilGEPS": {"primary": "#0e6655", "secondary": "#27ae60", "accent": "#58d68d", "bg": "#e8f6f3"},
}
EDA_CMAP_BAR = "viridis"  # Enhanced bar/hist colors
EDA_CMAP_HEATMAP = "RdYlBu_r"  # Correlation heatmap


def _get_data_source_colors(data_source: str | None) -> dict:
    """Return color dict for data source. Default to neutral if unknown."""
    if data_source and data_source.upper() == "DOH":
        return EDA_COLORS["DOH"]
    if data_source and data_source.upper() == "PHILGEPS":
        return EDA_COLORS["PhilGEPS"]
    return {"primary": "#2c3e50", "secondary": "#7f8c8d", "accent": "#bdc3c7", "bg": "#ecf0f1"}


def _add_data_source_badge(ax, data_source: str | None) -> None:
    """Add data source badge (DOH/PhilGEPS) to plot corner."""
    if not data_source:
        return
    src = str(data_source).upper()
    colors = _get_data_source_colors(data_source)
    ax.text(0.98, 0.98, f"Source: {src}", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="right", bbox=dict(boxstyle="round,pad=0.4",
            facecolor=colors["accent"], edgecolor=colors["primary"], alpha=0.9))


def _safe_filename(s: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if ch in bad else ch for ch in str(s))
    out = out.replace("\n", " ").strip()
    return out[:180] if len(out) > 180 else out


def _human_readable_label(name: str) -> str:
    """Convert variable names to human-readable labels (e.g., medicines_received -> Medicines Received)."""
    s = str(name).replace("_", " ").replace("-", " ")
    return s.strip().title() if s else name


def _plot_missingness_bar(df: pd.DataFrame, out_path: str, *, title: str, top_n: int = 20, data_source: str | None = None) -> None:
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0].head(top_n)
    if miss.empty:
        return
    colors = _get_data_source_colors(data_source)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160, facecolor=colors["bg"])
    bars = ax.barh(range(len(miss)), miss.values[::-1], color=colors["secondary"], edgecolor=colors["primary"], linewidth=1.2)
    ax.set_yticks(range(len(miss)))
    ax.set_yticklabels([_human_readable_label(str(x))[:45] for x in miss.index[::-1]], fontsize=10)
    ax.set_xlabel("Number of missing values", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_facecolor(colors["bg"])
    _add_data_source_badge(ax, data_source)
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def _plot_numeric_hists(
    df: pd.DataFrame,
    out_dir: str,
    *,
    title_prefix: str,
    cols: list[str] | None = None,
    data_source: str | None = None,
) -> None:
    _ensure_dir(out_dir)
    num = df.select_dtypes(include="number")
    if cols:
        cols = [c for c in cols if c in num.columns]
        num = num[cols] if cols else num
    if num.shape[1] == 0:
        return
    colors = _get_data_source_colors(data_source)
    for c in list(num.columns)[:20]:
        s = pd.to_numeric(num[c], errors="coerce").dropna()
        if s.empty:
            continue
        if len(s) > 50000:
            s = s.sample(50000, random_state=42)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160, facecolor=colors["bg"])
        ax.hist(s, bins=50, color=colors["secondary"], edgecolor=colors["primary"], alpha=0.85, linewidth=0.8)
        ax.set_title(f"{title_prefix} - Distribution of: {_human_readable_label(c)}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(_human_readable_label(c), fontsize=11)
        ax.set_ylabel("Number of records", fontsize=11)
        ax.set_facecolor(colors["bg"])
        _add_data_source_badge(ax, data_source)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_{_safe_filename(c)}.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        plt.close()


def _plot_categorical_top_bars(
    df: pd.DataFrame,
    out_dir: str,
    *,
    title_prefix: str,
    top_n_cols: int = 12,
    top_k: int = 15,
    data_source: str | None = None,
) -> None:
    _ensure_dir(out_dir)
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        return
    scored: list[tuple[int, str]] = []
    for c in obj_cols:
        nunique = int(df[c].nunique(dropna=True))
        if 2 <= nunique <= 200:
            scored.append((nunique, c))
    scored.sort(key=lambda x: x[0])
    chosen = [c for _, c in scored[:top_n_cols]]
    colors = _get_data_source_colors(data_source)
    for c in chosen:
        vc = df[c].fillna("N/A").astype(str).value_counts().head(top_k)
        if vc.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 7), dpi=160, facecolor=colors["bg"])
        bars = ax.barh(range(len(vc)), vc.values[::-1], color=colors["secondary"], edgecolor=colors["primary"], linewidth=1.0)
        ax.set_yticks(range(len(vc)))
        ax.set_yticklabels([_human_readable_label(str(x))[:50] for x in vc.index[::-1]], fontsize=10)
        ax.set_xlabel("Count (number of records)", fontsize=12, fontweight="bold")
        ax.set_title(f"{title_prefix} - Top {top_k} values for: {_human_readable_label(c)}", fontsize=12, fontweight="bold", pad=10)
        ax.set_facecolor(colors["bg"])
        _add_data_source_badge(ax, data_source)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cat_top_{_safe_filename(c)}.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        plt.close()


def _plot_correlation_heatmap(
    df: pd.DataFrame,
    out_path: str,
    *,
    title: str,
    cols: list[str] | None = None,
    max_cols: int = 15,
    data_source: str | None = None,
) -> None:
    num = df.select_dtypes(include="number")
    if cols:
        cols = [c for c in cols if c in num.columns]
        num = num[cols] if cols else num
    if num.shape[1] < 2:
        return
    num = num.iloc[:, :max_cols]
    corr = num.corr()
    if corr.empty or corr.shape[0] < 2:
        return
    colors = _get_data_source_colors(data_source)
    fig, ax = plt.subplots(figsize=(11, 9), dpi=160, facecolor=colors["bg"])
    im = ax.imshow(corr.values, cmap=EDA_CMAP_HEATMAP, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels([_human_readable_label(str(c))[:22] for c in corr.columns], rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels([_human_readable_label(str(c))[:22] for c in corr.columns], fontsize=10)
    cbar = plt.colorbar(im, ax=ax, label="Correlation (-1 to +1: negative to positive)", shrink=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_facecolor(colors["bg"])
    _add_data_source_badge(ax, data_source)
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def _plot_summary_table(
    df: pd.DataFrame,
    out_path: str,
    *,
    title: str,
    data_source: str | None = None,
) -> None:
    """Generate summary statistics table - readable for non-analysts."""
    if df is None or df.empty:
        return
    colors = _get_data_source_colors(data_source)
    num = df.select_dtypes(include="number")
    rows, cols = df.shape
    summary_data = [
        ["Number of rows", f"{rows:,}"],
        ["Number of columns", f"{cols:,}"],
    ]
    if not num.empty:
        try:
            desc = num.describe().T
            for idx in list(desc.index)[:8]:
                mean_val = desc.loc[idx, "mean"] if "mean" in desc.columns else 0
                std_val = desc.loc[idx, "std"] if "std" in desc.columns else 0
                label = _human_readable_label(str(idx)[:35])
                summary_data.append([label, f"Mean: {float(mean_val):.2f}  |  Std Dev: {float(std_val):.2f}"])
        except Exception:
            pass
    tab = pd.DataFrame(summary_data[:12], columns=["What", "Value"])
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160, facecolor="#f8f9fa")
    ax.axis("off")
    tbl = ax.table(
        cellText=tab.values,
        colLabels=tab.columns,
        loc="center",
        cellLoc="left",
        colColours=["#2c3e50", "#2c3e50"],
    )
    tbl.auto_set_font_size(False)
    tbl.scale(1.4, 2.6)
    for (i, j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_text_props(color="white", fontweight="bold", fontsize=13)
            cell.set_facecolor("#2c3e50")
        else:
            cell.set_text_props(color="#1a1a1a", fontweight="normal", fontsize=11)
            cell.set_facecolor(colors["bg"] if i % 2 == 1 else "#e8eef2")
    ax.set_title(f"{title}", fontsize=14, fontweight="bold", pad=16, color="#1a1a1a")
    _add_data_source_badge(ax, data_source)
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()


def generate_eda_visualizations(
    df: pd.DataFrame,
    *,
    out_dir: str,
    dataset_label: str,
    stage_label: str,
    numeric_focus: list[str] | None = None,
    include_correlation: bool = True,
    data_source: str | None = None,
) -> None:
    """Generate missingness, histograms, categorical bars, correlation heatmap, summary table.
    data_source: 'DOH' or 'PhilGEPS' - shown as badge on all plots."""
    if df is None or df.empty:
        return
    _ensure_dir(out_dir)
    title_prefix = f"{dataset_label} [{stage_label}]"
    _plot_missingness_bar(
        df, os.path.join(out_dir, "missingness_top.png"),
        title=f"{title_prefix} - Missing values (top columns)", top_n=25, data_source=data_source,
    )
    _plot_numeric_hists(df, os.path.join(out_dir, "numeric_hists"), title_prefix=title_prefix,
                       cols=numeric_focus, data_source=data_source)
    _plot_categorical_top_bars(df, os.path.join(out_dir, "categorical_top"), title_prefix=title_prefix,
                               data_source=data_source)
    if include_correlation and numeric_focus:
        _plot_correlation_heatmap(
            df, os.path.join(out_dir, "correlation_heatmap.png"),
            title=f"{title_prefix} - Numeric correlation", cols=numeric_focus, max_cols=12, data_source=data_source,
        )
    _plot_summary_table(df, os.path.join(out_dir, "summary_table.png"), title=f"{title_prefix} - Summary",
                       data_source=data_source)


def infer_time_label(name: str) -> str:
    s = str(name)
    digits = "".join(ch for ch in s if ch.isdigit())
    for i in range(max(0, len(digits) - 3)):
        chunk = digits[i : i + 4]
        if chunk.startswith(("19", "20")):
            return chunk
    return s


def profile_frame(df: pd.DataFrame, *, key_columns: list[str] | None, top_n: int = 10) -> dict:
    rows, cols = df.shape
    dtypes = df.dtypes.astype(str).to_dict()
    dup_count = int(df.duplicated().sum()) if rows else 0
    missing_keys = [c for c in (key_columns or []) if c not in df.columns]
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    missing_top = missing_counts.head(top_n).to_dict()
    return {
        "rows": rows, "cols": cols, "dtypes": dtypes,
        "dup_count": dup_count, "missing_keys": missing_keys, "missing_top": missing_top,
    }


def rank_attributes_for_clustering(
    df: pd.DataFrame,
    *,
    objective: str,
    must_have: list[str] | None = None,
    exclude_contains: list[str] | None = None,
) -> dict:
    rows = max(int(df.shape[0]), 1)
    exclude_contains = exclude_contains or []
    must_have = must_have or []
    numeric_cols = list(df.select_dtypes(include="number").columns)
    all_cols = list(df.columns)
    usable_numeric, needs_encoding, likely_ids, drop_or_review = [], [], [], []
    for c in all_cols:
        c_str = str(c)
        ser = df[c]
        miss_pct = float(ser.isna().mean() * 100.0)
        nunique = int(ser.nunique(dropna=True))
        if any(token.lower() in c_str.lower() for token in exclude_contains):
            drop_or_review.append((c_str, "excluded_by_name_rule"))
            continue
        if nunique / rows > 0.8 and rows >= 50 and c_str not in must_have and ser.dtype == object:
            likely_ids.append((c_str, f"high_cardinality ({nunique} unique / {rows} rows)"))
            continue
        if c_str in numeric_cols:
            if miss_pct < 60:
                usable_numeric.append((c_str, f"numeric, missing={miss_pct:.1f}%"))
            else:
                drop_or_review.append((c_str, f"numeric but very missing ({miss_pct:.1f}%)"))
        else:
            if miss_pct < 60 and nunique <= 200:
                needs_encoding.append((c_str, f"categorical/mixed, nunique={nunique}, missing={miss_pct:.1f}%"))
            else:
                drop_or_review.append((c_str, f"categorical/mixed needs review (nunique={nunique}, missing={miss_pct:.1f}%)"))
    for c in must_have:
        if c in df.columns and c not in [x[0] for x in usable_numeric] and c not in [x[0] for x in needs_encoding]:
            ser = df[c]
            miss_pct = float(ser.isna().mean() * 100.0)
            nunique = int(ser.nunique(dropna=True))
            needs_encoding.append((c, f"must-have; dtype={ser.dtype}, nunique={nunique}, missing={miss_pct:.1f}%"))
    return {"objective": objective, "usable_numeric": usable_numeric, "needs_encoding": needs_encoding, "likely_ids": likely_ids, "drop_or_review": drop_or_review}


def write_snapshots_block(
    *,
    dataset_name: str,
    stage: str,
    snapshots: list[dict],
    key_columns: list[str] | None,
    objective: str,
    merged_df: pd.DataFrame | None,
    ranking: dict | None,
    log_fn,
) -> None:
    def section(title: str) -> None:
        log_fn("")
        log_fn("=" * 80)
        log_fn(title.replace("—", "-").replace("–", "-"))
        log_fn("=" * 80)
    section(f"{dataset_name} - {stage} - Data Understanding Snapshots")
    log_fn(f"- Objective: {objective}")
    if key_columns:
        log_fn(f"- Expected key columns: {key_columns}")
    log_fn(f"- Snapshots: {len(snapshots)}")
    for s in snapshots:
        label = s["label"]
        prof = s["profile"]
        log_fn("")
        log_fn("-" * 80)
        log_fn(f"Snapshot: {label}")
        log_fn(f"Rows, Cols: {prof['rows']}, {prof['cols']}")
        if prof["missing_keys"]:
            log_fn(f"Missing key columns: {prof['missing_keys']}")
        log_fn(f"Duplicate rows: {prof['dup_count']}")
        if prof["missing_top"]:
            for k, v in prof["missing_top"].items():
                log_fn(f"  - {k}: {int(v)}")
    if merged_df is None or ranking is None:
        return
    section(f"{dataset_name} - {stage} - Attribute selection (K-Means + DBSCAN)")
    log_fn(f"- Objective: {ranking['objective']}")
    if ranking["usable_numeric"]:
        for col, why in ranking["usable_numeric"][:30]:
            log_fn(f"  - {col}: {why}")
    if ranking["needs_encoding"]:
        for col, why in ranking["needs_encoding"][:30]:
            log_fn(f"  - {col}: {why}")


def write_snapshot(df: pd.DataFrame, label: str, *, key_columns: list[str] | None, top_n: int, log_fn) -> None:
    prof = profile_frame(df, key_columns=key_columns, top_n=top_n)
    log_fn("")
    log_fn("=" * 80)
    log_fn(f"DATA UNDERSTANDING SNAPSHOT: {label}")
    log_fn("=" * 80)
    log_fn(f"Rows, Cols: {prof['rows']}, {prof['cols']}")
    if prof["missing_keys"]:
        log_fn(f"Missing key columns: {prof['missing_keys']}")
    log_fn(f"Duplicate rows: {prof['dup_count']}")
    if prof["missing_top"]:
        for k, v in prof["missing_top"].items():
            log_fn(f"  - {k}: {int(v)}")


def build_visualization_paths(project_root: str) -> dict:
    """Step: Build paths for EDA outputs (01_data_cleaning/steps, logs, etc.)."""
    vis_root = os.path.join(project_root, "webp")
    logs_dir = os.path.join(vis_root, "logs")
    ev_dir = os.path.join(vis_root, "EDA_and_visualization")
    step01_dir = os.path.join(ev_dir, "01_data_cleaning")
    du_txt_dir = os.path.join(step01_dir, "data_understanding")
    eda_steps_dir = os.path.join(step01_dir, "steps")
    step02_dir = os.path.join(ev_dir, "02_data_transformation")
    eda_preprocessing_steps_dir = os.path.join(step02_dir, "steps")
    for d in (logs_dir, du_txt_dir, step01_dir, eda_steps_dir, step02_dir, eda_preprocessing_steps_dir):
        _ensure_dir(d)
    return {
        "vis_root_dir": vis_root, "vis_du_txt_dir": du_txt_dir, "vis_eda_steps_dir": eda_steps_dir,
        "vis_eda_preprocessing_dir": step02_dir, "vis_eda_preprocessing_steps_dir": eda_preprocessing_steps_dir,
        "vis_prep_dir": logs_dir,
    }


@dataclass
class LogSinks:
    prep_log_path: str
    du_before_log_path: str
    du_after_log_path: str
    _prep_fh: object
    _du_before_fh: object
    _du_after_fh: object

    def close(self) -> None:
        for fh in (self._prep_fh, self._du_before_fh, self._du_after_fh):
            try:
                fh.close()
            except Exception:
                pass

    def log_prep(self, message: str = "") -> None:
        self._prep_fh.write(f"{message}\n")
        self._prep_fh.flush()

    def log_du_before(self, message: str = "") -> None:
        self._du_before_fh.write(f"{message}\n")
        self._du_before_fh.flush()

    def log_du_after(self, message: str = "") -> None:
        self._du_after_fh.write(f"{message}\n")
        self._du_after_fh.flush()


def open_log_sinks(*, prep_log_path: str, du_before_log_path: str, du_after_log_path: str) -> LogSinks:
    prep_fh = open(prep_log_path, "w", encoding="utf-8")
    du_before_fh = open(du_before_log_path, "w", encoding="utf-8")
    du_after_fh = open(du_after_log_path, "w", encoding="utf-8")
    sinks = LogSinks(
        prep_log_path=prep_log_path, du_before_log_path=du_before_log_path, du_after_log_path=du_after_log_path,
        _prep_fh=prep_fh, _du_before_fh=du_before_fh, _du_after_fh=du_after_fh,
    )
    atexit.register(sinks.close)
    return sinks


# -----------------------------------------------------------------------------
# Step: Clustering config - dataset paths (minmax), id columns for A, B, C
# -----------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    name: str
    path_minmax: str
    id_columns: list[str]


def get_dataset_configs() -> list[DatasetConfig]:
    root = _project_root()
    trans_dir = os.path.join(root, "this_datasets", "02_data_transformation")
    return [
        DatasetConfig(
            name="A_supplier_awardee",
            path_minmax=os.path.join(trans_dir, "clustering_A_supplier_awardee_features_minmax.csv"),
            id_columns=["Awardee Organization Name"],
        ),
        DatasetConfig(
            name="B_medicine_procurement_pattern",
            path_minmax=os.path.join(trans_dir, "clustering_B_medicine_procurement_pattern_features_minmax.csv"),
            id_columns=[
                "Awardee Organization Name", "Item Name", "UNSPSC Description", "Region of Awardee",
                "Procurement Mode", "Funding Source", "Award Reference No.", "UNSPSC Code",
            ],
        ),
        DatasetConfig(
            name="C_distribution_recipient",
            path_minmax=os.path.join(trans_dir, "clustering_C_distribution_recipient_features_minmax.csv"),
            id_columns=["RECIPIENT", "top_program", "top_item_description"],
        ),
    ]


# -----------------------------------------------------------------------------
# Step: Transformation helpers - coerce numeric, fill categorical, scale, outliers
# -----------------------------------------------------------------------------
def _coerce_numeric_and_impute(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(med if pd.notna(med) else 0)
    return df


def _fill_categorical(df: pd.DataFrame, cols: list[str], placeholder: str = "N/A") -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = df[c].replace({None: np.nan, "nan": np.nan, "None": np.nan})
        df[c] = df[c].fillna(placeholder).astype(str)
    return df


def scale_numeric_features(df: pd.DataFrame, *, feature_cols: list[str], method: str) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in feature_cols if c in out.columns]
    if not cols:
        return out
    X = out[cols].to_numpy(dtype=float)
    scaler = StandardScaler() if method == "zscore" else MinMaxScaler()
    out[cols] = scaler.fit_transform(X)
    return out


def analyze_outliers_iqr_and_boxplots(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    dataset_label: str,
    out_dir: str,
    anomalies_out_csv: str,
    summary_out_csv: str,
    data_source: str | None = None,
) -> None:
    df = df.copy()
    for c in feature_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        return
    _ensure_dir(out_dir)
    colors = _get_data_source_colors(data_source)
    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160, facecolor=colors["bg"])
        bp = ax.boxplot(s, vert=True, patch_artist=True)
        bp["boxes"][0].set_facecolor(colors["secondary"])
        bp["boxes"][0].set_alpha(0.8)
        bp["medians"][0].set_color(colors["primary"])
        bp["medians"][0].set_linewidth(2)
        ax.set_title(f"{dataset_label} - Distribution and outliers: {_human_readable_label(c)}", fontsize=12, fontweight="bold")
        ax.set_ylabel(_human_readable_label(c), fontsize=11)
        ax.set_facecolor(colors["bg"])
        _add_data_source_badge(ax, data_source)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"boxplot__{c}.png"), facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        plt.close()
    outlier_masks = {}
    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_masks[c] = ((pd.to_numeric(df[c], errors="coerce") < lo) | (pd.to_numeric(df[c], errors="coerce") > hi)).fillna(False)
    if not outlier_masks:
        return
    is_anomaly = pd.concat(outlier_masks.values(), axis=1).any(axis=1)
    df[is_anomaly].to_csv(os.path.join(out_dir, anomalies_out_csv), index=False)
    summary_rows = [{"feature_col": c, "outlier_count": int(m.sum()), "outlier_ratio": float(m.mean())} for c, m in outlier_masks.items()]
    summary_rows.append({"feature_col": "__ANY__", "outlier_count": int(is_anomaly.sum()), "outlier_ratio": float(is_anomaly.mean())})
    pd.DataFrame(summary_rows).sort_values("outlier_count", ascending=False).to_csv(os.path.join(out_dir, summary_out_csv), index=False)


def load_features(cfg: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray] | None:
    """Step: Load minmax-scaled features; return (full_df, id_df, X) for clustering. Returns None if missing."""
    if not os.path.exists(cfg.path_minmax) or os.path.getsize(cfg.path_minmax) == 0:
        return None
    df = pd.read_csv(cfg.path_minmax, low_memory=False)
    if df.empty:
        return None
    id_cols = [c for c in cfg.id_columns if c in df.columns]
    feature_cols = [c for c in df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        return None
    id_df = df[id_cols].copy() if id_cols else pd.DataFrame()
    X = df[feature_cols].to_numpy(dtype=float)
    return df, id_df, X
