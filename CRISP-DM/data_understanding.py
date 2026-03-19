from __future__ import annotations

import atexit
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import pandas as pd


def _safe_filename(s: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if ch in bad else ch for ch in str(s))
    out = out.replace("\n", " ").strip()
    return out[:180] if len(out) > 180 else out


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _plot_missingness_bar(
    df: pd.DataFrame,
    out_path: str,
    *,
    title: str,
    top_n: int = 20,
) -> None:
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0].head(top_n)
    if miss.empty:
        return
    plt.figure(figsize=(12, 6), dpi=160)
    miss.iloc[::-1].plot(kind="barh")
    plt.title(title)
    plt.xlabel("Missing values (count)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_numeric_hists(
    df: pd.DataFrame,
    out_dir: str,
    *,
    title_prefix: str,
    cols: list[str] | None = None,
) -> None:
    _ensure_dir(out_dir)
    num = df.select_dtypes(include="number")
    if cols:
        cols = [c for c in cols if c in num.columns]
        num = num[cols] if cols else num
    if num.shape[1] == 0:
        return

    for c in list(num.columns)[:20]:
        s = pd.to_numeric(num[c], errors="coerce").dropna()
        if s.empty:
            continue
        if len(s) > 50000:
            s = s.sample(50000, random_state=42)

        plt.figure(figsize=(10, 5), dpi=160)
        plt.hist(s, bins=50)
        plt.title(f"{title_prefix} - Histogram: {c}")
        plt.xlabel(c)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_{_safe_filename(c)}.png"))
        plt.close()


def _plot_categorical_top_bars(
    df: pd.DataFrame,
    out_dir: str,
    *,
    title_prefix: str,
    top_n_cols: int = 12,
    top_k: int = 15,
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

    for c in chosen:
        vc = df[c].fillna("N/A").astype(str).value_counts().head(top_k)
        if vc.empty:
            continue
        plt.figure(figsize=(12, 6), dpi=160)
        vc.iloc[::-1].plot(kind="barh")
        plt.title(f"{title_prefix} - Top {top_k} categories: {c}")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cat_top_{_safe_filename(c)}.png"))
        plt.close()


def generate_eda_visualizations(
    df: pd.DataFrame,
    *,
    out_dir: str,
    dataset_label: str,
    stage_label: str,
    numeric_focus: list[str] | None = None,
) -> None:
    """
    Data-analyst style EDA visuals:
    - Missingness bar chart (top columns)
    - Numeric histograms
    - Top-category bar charts (categorical distributions)
    """
    if df is None or df.empty:
        return

    _ensure_dir(out_dir)
    title_prefix = f"{dataset_label} [{stage_label}]"

    _plot_missingness_bar(
        df,
        os.path.join(out_dir, "missingness_top.png"),
        title=f"{title_prefix} - Missing values (top columns)",
        top_n=25,
    )
    _plot_numeric_hists(
        df,
        os.path.join(out_dir, "numeric_hists"),
        title_prefix=title_prefix,
        cols=numeric_focus,
    )
    _plot_categorical_top_bars(
        df,
        os.path.join(out_dir, "categorical_top"),
        title_prefix=title_prefix,
        top_n_cols=12,
        top_k=15,
    )


def section(log_fn, title: str) -> None:
    log_fn("")
    log_fn("=" * 100)
    log_fn(title.replace("—", "-").replace("–", "-"))
    log_fn("=" * 100)


def infer_time_label(name: str) -> str:
    s = str(name)
    digits = "".join(ch for ch in s if ch.isdigit())
    for i in range(0, len(digits) - 3):
        chunk = digits[i : i + 4]
        if chunk.startswith(("19", "20")):
            return chunk
    return s


def profile_frame(df: pd.DataFrame, *, key_columns: list[str] | None, top_n: int = 10) -> dict:
    rows, cols = df.shape
    dtypes = df.dtypes.astype(str).to_dict()
    dup_count = int(df.duplicated().sum()) if rows else 0

    missing_keys: list[str] = []
    if key_columns:
        missing_keys = [c for c in key_columns if c not in df.columns]

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    missing_top = missing_counts.head(top_n).to_dict()

    return {
        "rows": rows,
        "cols": cols,
        "dtypes": dtypes,
        "dup_count": dup_count,
        "missing_keys": missing_keys,
        "missing_top": missing_top,
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

    usable_numeric = []
    needs_encoding = []
    likely_ids = []
    drop_or_review = []

    for c in all_cols:
        c_str = str(c)
        ser = df[c]
        miss_pct = float(ser.isna().mean() * 100.0)
        nunique = int(ser.nunique(dropna=True))

        if any(token.lower() in c_str.lower() for token in exclude_contains):
            drop_or_review.append((c_str, "excluded_by_name_rule"))
            continue

        high_card = nunique / rows > 0.8 and rows >= 50
        if high_card and c_str not in must_have and ser.dtype == object:
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
                drop_or_review.append(
                    (c_str, f"categorical/mixed needs review (nunique={nunique}, missing={miss_pct:.1f}%)")
                )

    for c in must_have:
        if c in df.columns and c not in [x[0] for x in usable_numeric] and c not in [x[0] for x in needs_encoding]:
            ser = df[c]
            miss_pct = float(ser.isna().mean() * 100.0)
            nunique = int(ser.nunique(dropna=True))
            needs_encoding.append((c, f"must-have; dtype={ser.dtype}, nunique={nunique}, missing={miss_pct:.1f}%"))

    return {
        "objective": objective,
        "usable_numeric": usable_numeric,
        "needs_encoding": needs_encoding,
        "likely_ids": likely_ids,
        "drop_or_review": drop_or_review,
    }


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
    section(log_fn, f"{dataset_name} - {stage} - Data Understanding Snapshots (consolidated)")
    log_fn(f"- Objective: {objective}")
    if key_columns:
        log_fn(f"- Expected key columns: {key_columns}")
    log_fn(f"- Snapshots included: {len(snapshots)}")

    for s in snapshots:
        label = s["label"]
        prof = s["profile"]
        log_fn("")
        log_fn("-" * 100)
        log_fn(f"Snapshot: {label}")
        log_fn(f"Rows, Cols: {prof['rows']}, {prof['cols']}")
        if prof["missing_keys"]:
            log_fn(f"Missing key columns: {prof['missing_keys']}")
        log_fn(f"Duplicate rows: {prof['dup_count']}")
        if prof["missing_top"]:
            log_fn("Missing values (top):")
            for k, v in prof["missing_top"].items():
                log_fn(f"  - {k}: {int(v)}")
        else:
            log_fn("Missing values: none")

    section(log_fn, f"{dataset_name} - {stage} - Attribute selection notes (K-Means + DBSCAN)")
    log_fn("Goal: identify meaningful attributes for similarity patterns and anomaly detection.")
    log_fn("Reminder: K-Means/DBSCAN need numeric feature vectors; categoricals require encoding.")
    log_fn("")
    log_fn("Recommended attribute triage (based on merged view of this stage):")

    if merged_df is None or ranking is None:
        log_fn("  (No merged view available at this stage.)")
        return

    log_fn(f"- Objective: {ranking['objective']}")
    if ranking["usable_numeric"]:
        log_fn("- Usable numeric (candidate features):")
        for col, why in ranking["usable_numeric"][:30]:
            log_fn(f"  - {col}: {why}")
    else:
        log_fn("- Usable numeric: none detected")

    if ranking["needs_encoding"]:
        log_fn("- Categorical/mixed (use after encoding/aggregation):")
        for col, why in ranking["needs_encoding"][:30]:
            log_fn(f"  - {col}: {why}")

    if ranking["likely_ids"]:
        log_fn("- Likely identifiers/free-text (usually exclude from clustering features):")
        for col, why in ranking["likely_ids"][:30]:
            log_fn(f"  - {col}: {why}")

    if ranking["drop_or_review"]:
        log_fn("- Drop or review (too missing / too high-cardinality / noisy):")
        for col, why in ranking["drop_or_review"][:30]:
            log_fn(f"  - {col}: {why}")


def write_snapshot(
    df: pd.DataFrame,
    label: str,
    *,
    key_columns: list[str] | None,
    top_n: int,
    log_fn,
) -> None:
    section(log_fn, f"DATA UNDERSTANDING SNAPSHOT: {label}")
    prof = profile_frame(df, key_columns=key_columns, top_n=top_n)
    log_fn(f"Rows, Cols: {prof['rows']}, {prof['cols']}")
    dtypes = prof["dtypes"]
    if len(dtypes) <= 60:
        log_fn(f"Dtypes: {dtypes}")
    else:
        log_fn(f"Dtypes: {len(dtypes)} columns (too many to print)")
    if prof["missing_keys"]:
        log_fn(f"Missing key columns: {prof['missing_keys']}")
    log_fn(f"Duplicate rows: {prof['dup_count']}")
    if prof["missing_top"]:
        log_fn(f"Missing values (top {top_n}):")
        for k, v in prof["missing_top"].items():
            log_fn(f"  - {k}: {int(v)}")
    else:
        log_fn("Missing values: none")

    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] > 0:
        desc = num_df.describe().T
        log_fn("Numeric summary (first columns):")
        log_fn(desc.head(min(top_n, len(desc))).to_string())
    else:
        log_fn("Numeric summary: no numeric columns detected")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_visualization_paths(project_root: str) -> dict:
    """
    Current visualization folder layout:
    - webp/logs/data_preparation_run.txt
    - webp/EDA/data_preparation/reports_txt/data_understanding_before.txt
    - webp/EDA/data_preparation/reports_txt/data_understanding_after.txt
    """
    vis_root = os.path.join(project_root, "webp")
    logs_dir = os.path.join(vis_root, "logs")

    eda_root_dir = os.path.join(vis_root, "EDA", "data_preparation")
    du_txt_dir = os.path.join(eda_root_dir, "reports_txt")
    du_eda_dir = eda_root_dir  # caller appends /before and /after

    ensure_dir(logs_dir)
    ensure_dir(du_txt_dir)
    ensure_dir(du_eda_dir)

    return {
        "vis_root_dir": vis_root,
        "vis_du_txt_dir": du_txt_dir,
        "vis_du_eda_dir": du_eda_dir,
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
        prep_log_path=prep_log_path,
        du_before_log_path=du_before_log_path,
        du_after_log_path=du_after_log_path,
        _prep_fh=prep_fh,
        _du_before_fh=du_before_fh,
        _du_after_fh=du_after_fh,
    )
    atexit.register(sinks.close)
    return sinks

