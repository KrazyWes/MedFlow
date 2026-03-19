import os
import pandas as pd
import matplotlib.pyplot as plt  # type: ignore[import-not-found]


def _safe_filename(s: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if ch in bad else ch for ch in str(s))
    out = out.replace("\n", " ").strip()
    return out[:180] if len(out) > 180 else out


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _plot_missingness_bar(df: pd.DataFrame, out_path: str, *, title: str, top_n: int = 20) -> None:
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


def _plot_numeric_hists(df: pd.DataFrame, out_dir: str, *, title_prefix: str, cols: list[str] | None = None) -> None:
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

