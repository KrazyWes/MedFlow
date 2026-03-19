import pandas as pd


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
                drop_or_review.append((c_str, f"categorical/mixed needs review (nunique={nunique}, missing={miss_pct:.1f}%)"))

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

