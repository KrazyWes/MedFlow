"""
Cluster Interpretation - MedFlow

Generates per-cluster summaries and profile reports for Objectives A, B, and C.
Outputs: cluster profiles printed to console and saved to model_outputs/cluster_reports/.

Usage: python CRISP-DM/interpret_clusters.py
"""

import os
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
outputs_dir = os.path.join(project_root, "model_outputs")
reports_dir = os.path.join(outputs_dir, "cluster_reports")
os.makedirs(reports_dir, exist_ok=True)


def safe_read_csv(path):
    """Load CSV if exists, else return None."""
    if os.path.isfile(path):
        return pd.read_csv(path, low_memory=False)
    return None


def format_num(x):
    """Format large numbers for display."""
    if pd.isna(x):
        return "N/A"
    if x >= 1_000_000:
        return f"{x/1e6:.2f}M"
    if x >= 1_000:
        return f"{x/1e3:.1f}K"
    return f"{x:.2f}"


# =============================================================================
# Objective A: Supplier Clusters
# =============================================================================
def interpret_supplier_clusters():
    path = os.path.join(outputs_dir, "philgeps_supplier_clustered.csv")
    df = safe_read_csv(path)
    if df is None or "cluster" not in df.columns:
        print("  [SKIP] philgeps_supplier_clustered.csv not found")
        return

    lines = ["=" * 60, "OBJECTIVE A: Supplier / Awardee Clusters", "=" * 60]

    # Summary table
    numeric_cols = ["total_contract_amount", "num_awards"]
    summary = df.groupby("cluster").agg(
        count=(numeric_cols[0], "count"),
        avg_contract_amount=(numeric_cols[0], "mean"),
        median_contract_amount=(numeric_cols[0], "median"),
        avg_num_awards=(numeric_cols[1], "mean"),
    ).round(2)

    lines.append("\nCluster size & metrics:")
    lines.append(summary.to_string())

    # Per-cluster profiles
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        lines.append(f"\n--- Cluster {c} (n={len(sub)}) ---")
        lines.append(f"  Avg total contract amount: {format_num(sub['total_contract_amount'].mean())}")
        lines.append(f"  Avg num awards:           {sub['num_awards'].mean():.1f}")
        if "Region of Awardee" in sub.columns:
            top_regions = sub["Region of Awardee"].value_counts().head(3)
            lines.append(f"  Top regions: {dict(top_regions)}")
        if "Procurement Mode" in sub.columns:
            top_mode = sub["Procurement Mode"].value_counts().iloc[0]
            lines.append(f"  Most common procurement: {sub['Procurement Mode'].mode().iloc[0]} ({top_mode} suppliers)")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(reports_dir, "supplier_cluster_profiles.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    summary.to_csv(os.path.join(reports_dir, "supplier_cluster_summary.csv"))
    print(f"\n  Saved: {reports_dir}/supplier_cluster_profiles.txt, supplier_cluster_summary.csv")


# =============================================================================
# Objective B: Procurement Pattern Clusters
# =============================================================================
def interpret_procurement_clusters():
    path = os.path.join(outputs_dir, "philgeps_procurement_clustered.csv")
    df = safe_read_csv(path)
    if df is None or "cluster" not in df.columns:
        print("  [SKIP] philgeps_procurement_clustered.csv not found")
        return

    lines = ["=" * 60, "OBJECTIVE B: Medicine Procurement Pattern Clusters", "=" * 60]

    numeric_cols = ["Item Budget", "Quantity"]
    summary = df.groupby("cluster").agg(
        count=("Item Budget", "count"),
        avg_item_budget=("Item Budget", "mean"),
        median_item_budget=("Item Budget", "median"),
        avg_quantity=("Quantity", "mean"),
    ).round(2)

    lines.append("\nCluster size & metrics:")
    lines.append(summary.to_string())

    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        lines.append(f"\n--- Cluster {c} (n={len(sub)}) ---")
        lines.append(f"  Avg item budget: {format_num(sub['Item Budget'].mean())}")
        lines.append(f"  Avg quantity:    {format_num(sub['Quantity'].mean())}")
        if "Procurement Mode" in sub.columns:
            top_mode = sub["Procurement Mode"].value_counts().iloc[0]
            lines.append(f"  Top procurement mode: {sub['Procurement Mode'].mode().iloc[0]} ({top_mode} records)")
        if "Funding Source" in sub.columns:
            top_fund = sub["Funding Source"].value_counts().iloc[0]
            lines.append(f"  Top funding: {sub['Funding Source'].mode().iloc[0]} ({top_fund} records)")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(reports_dir, "procurement_cluster_profiles.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    summary.to_csv(os.path.join(reports_dir, "procurement_cluster_summary.csv"))
    print(f"\n  Saved: {reports_dir}/procurement_cluster_profiles.txt, procurement_cluster_summary.csv")


# =============================================================================
# Objective C: Distribution Clusters (DOH)
# =============================================================================
def interpret_distribution_clusters():
    path = os.path.join(outputs_dir, "doh_distribution_clustered.csv")
    df = safe_read_csv(path)
    if df is None or "cluster" not in df.columns:
        print("  [SKIP] doh_distribution_clustered.csv not found (run modeling.py for Objective C)")
        return

    lines = ["=" * 60, "OBJECTIVE C: Distribution Pattern Clusters (DOH)", "=" * 60]

    agg_cols = ["total_quantity", "total_amount", "num_distinct_medicines"]
    if "delivery_frequency" in df.columns:
        agg_cols.append("delivery_frequency")

    summary = df.groupby("cluster").agg(
        count=("RECIPIENT", "count"),
        avg_total_amount=("total_amount", "mean"),
        avg_quantity=("total_quantity", "mean"),
        avg_medicines=("num_distinct_medicines", "mean"),
    ).round(2)
    if "delivery_frequency" in df.columns:
        summary["avg_delivery_freq"] = df.groupby("cluster")["delivery_frequency"].mean().round(2)

    lines.append("\nCluster size & metrics:")
    lines.append(summary.to_string())

    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        lines.append(f"\n--- Cluster {c} (n={len(sub)}) ---")
        lines.append(f"  Avg total amount:      {format_num(sub['total_amount'].mean())}")
        lines.append(f"  Avg quantity:          {format_num(sub['total_quantity'].mean())}")
        lines.append(f"  Avg distinct medicines: {sub['num_distinct_medicines'].mean():.1f}")
        if "delivery_frequency" in sub.columns:
            lines.append(f"  Avg delivery frequency: {sub['delivery_frequency'].mean():.1f}")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(reports_dir, "distribution_cluster_profiles.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    summary.to_csv(os.path.join(reports_dir, "distribution_cluster_summary.csv"))
    print(f"\n  Saved: {reports_dir}/distribution_cluster_profiles.txt, distribution_cluster_summary.csv")


# =============================================================================
# Main
# =============================================================================
def main():
    print("Cluster interpretation - MedFlow\n")
    interpret_supplier_clusters()
    print("\n")
    interpret_procurement_clusters()
    print("\n")
    interpret_distribution_clusters()
    print("\nDone. Reports saved to model_outputs/cluster_reports/")


if __name__ == "__main__":
    main()
