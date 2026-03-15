"""
Cluster Visualizations - MedFlow

Generates figures for K-means clustering results:
  - Cluster size bar charts (A, B, C)
  - PCA 2D scatter plots colored by cluster

Saves figures to model_outputs/figures/.

Usage: python CRISP-DM/visualize_clusters.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =============================================================================
# Configuration
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
outputs_dir = os.path.join(project_root, "model_outputs")
prep_dir = os.path.join(project_root, "model_prep_outputs")
figures_dir = os.path.join(outputs_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# For large datasets, subsample for scatter plot (None = use all)
SCATTER_SUBSAMPLE = 5000  # procurement has 53k rows
RANDOM_STATE = 42
plt.rcParams["figure.dpi"] = 120


def safe_load(objective):
    """Load clustered CSV and X array for an objective. Returns (df, X) or (None, None)."""
    if objective == "A":
        csv_name, x_name = "philgeps_supplier_clustered.csv", "philgeps_supplier_X.npy"
    elif objective == "B":
        csv_name, x_name = "philgeps_procurement_clustered.csv", "philgeps_procurement_X.npy"
    else:
        csv_name, x_name = "doh_distribution_clustered.csv", "doh_distribution_X.npy"

    csv_path = os.path.join(outputs_dir, csv_name)
    X_path = os.path.join(prep_dir, x_name)

    if not os.path.isfile(csv_path):
        return None, None
    df = pd.read_csv(csv_path, low_memory=False)
    if "cluster" not in df.columns:
        return None, None
    if not os.path.isfile(X_path) or len(np.load(X_path)) != len(df):
        return df, None
    return df, np.load(X_path)


def plot_cluster_sizes(df, title, filename):
    """Bar chart of counts per cluster."""
    counts = df["cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.index.astype(str), counts.values, color=plt.cm.Set3(np.linspace(0, 1, len(counts))))
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.set_title(title)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(counts) * 0.01,
                str(int(b.get_height())), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_pca_scatter(df, X, title, filename, subsample=None):
    """2D PCA scatter colored by cluster."""
    if X is None or X.shape[1] < 2:
        return
    n = len(X)
    if subsample and n > subsample:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(n, subsample, replace=False)
        X_plot = X[idx]
        labels = df["cluster"].iloc[idx].values
    else:
        X_plot = X
        labels = df["cluster"].values

    n_components = min(2, X_plot.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(X_plot)

    fig, ax = plt.subplots(figsize=(8, 6))
    clusters = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(clusters), 10)))
    for i, c in enumerate(clusters):
        mask = labels == c
        ax.scatter(X2[mask, 0], X2[mask, 1], c=[colors[i]], label=f"Cluster {c}", alpha=0.5, s=15)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)" if n_components >= 2 else "PC2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def main():
    print("Generating cluster visualizations...\n")

    # --- Objective A: Supplier ---
    df_a, X_a = safe_load("A")
    if df_a is not None:
        plot_cluster_sizes(df_a, "Objective A: Supplier / Awardee Cluster Sizes", "supplier_cluster_sizes.png")
        plot_pca_scatter(df_a, X_a, "Objective A: Suppliers (PCA)", "supplier_pca_scatter.png")

    # --- Objective B: Procurement ---
    df_b, X_b = safe_load("B")
    if df_b is not None:
        plot_cluster_sizes(df_b, "Objective B: Procurement Pattern Cluster Sizes", "procurement_cluster_sizes.png")
        plot_pca_scatter(df_b, X_b, "Objective B: Procurement (PCA, subsampled)", "procurement_pca_scatter.png",
                        subsample=SCATTER_SUBSAMPLE)

    # --- Objective C: Distribution ---
    df_c, X_c = safe_load("C")
    if df_c is not None:
        plot_cluster_sizes(df_c, "Objective C: Distribution Cluster Sizes", "distribution_cluster_sizes.png")
        plot_pca_scatter(df_c, X_c, "Objective C: Distribution (PCA)", "distribution_pca_scatter.png")

    print(f"\nFigures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
