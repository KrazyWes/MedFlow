"""
Modeling - K-means Clustering for PhilGEPS and DOH Medical Data

Runs K-means on prepared features for:
  A. Supplier / Awardee Clustering (PhilGEPS)
  B. Medicine Procurement Pattern Clustering (PhilGEPS)
  C. Distribution Pattern Clustering (DOH)

Uses elbow method and silhouette score to evaluate cluster counts.
Outputs: models, labels, and clustered datasets in model_outputs/.

Usage: python CRISP-DM/modeling.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =============================================================================
# Configuration
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
prep_dir = os.path.join(project_root, "model_prep_outputs")
output_dir = os.path.join(project_root, "model_outputs")
os.makedirs(output_dir, exist_ok=True)

RANDOM_STATE = 42
K_RANGE = range(2, 11)  # Test k from 2 to 10
DEFAULT_K_A = 5   # Supplier clusters (can override via elbow/silhouette)
DEFAULT_K_B = 6   # Procurement clusters
DEFAULT_K_C = 5   # DOH distribution clusters


# =============================================================================
# Load Prepared Data
# =============================================================================
def load_prepared(objective: str):
    """Load X array and metadata for objective 'A', 'B', or 'C'."""
    if objective == "A":
        prefix, csv_name = "philgeps_supplier", "philgeps_supplier_aggregated.csv"
    elif objective == "B":
        prefix, csv_name = "philgeps_procurement", "philgeps_procurement_line_items.csv"
    elif objective == "C":
        prefix, csv_name = "doh_distribution", "doh_distribution_aggregated.csv"
    else:
        raise ValueError(f"Unknown objective: {objective}")

    X_path = os.path.join(prep_dir, f"{prefix}_X.npy")
    meta_path = os.path.join(prep_dir, f"{prefix}_meta.pkl")
    csv_path = os.path.join(prep_dir, csv_name)

    if not os.path.isfile(X_path):
        raise FileNotFoundError(f"Run modeling_preparation.py first. Expected: {X_path}")

    X = np.load(X_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    df = pd.read_csv(csv_path, low_memory=False)
    return X, meta, df


# =============================================================================
# Elbow & Silhouette
# =============================================================================
def evaluate_cluster_counts(X, k_range=None):
    """Compute inertia and silhouette for each k. Returns (inertias, silhouettes)."""
    k_range = k_range or K_RANGE
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        if k >= 2:
            silhouettes.append(silhouette_score(X, labels))
        else:
            silhouettes.append(np.nan)

    return inertias, silhouettes


def find_best_k_elbow(inertias, k_range=None):
    """Simple elbow: pick k where rate of decrease slows. Returns suggested k."""
    k_range = list(k_range or K_RANGE)
    if len(inertias) < 3:
        return k_range[0]
    diffs = np.diff(inertias)
    # Elbow: where second derivative is small (curvature flattens)
    curv = np.abs(np.diff(diffs))
    idx = np.argmin(curv)
    return k_range[idx + 1]


def find_best_k_silhouette(silhouettes, k_range=None):
    """Pick k with highest silhouette score."""
    k_range = list(k_range or K_RANGE)
    valid = [(k, s) for k, s in zip(k_range, silhouettes) if not np.isnan(s)]
    if not valid:
        return k_range[0]
    return max(valid, key=lambda x: x[1])[0]


# =============================================================================
# Run K-means and Save
# =============================================================================
def run_kmeans_and_save(objective: str, n_clusters: int, use_suggested: bool = False):
    """
    Load data, optionally find best k, run K-means, save model and labeled data.
    use_suggested: if True, run elbow/silhouette and use suggested k
    """
    X, meta, df = load_prepared(objective)
    k_range = list(K_RANGE)

    if use_suggested:
        print(f"  Evaluating k in {k_range}...")
        inertias, silhouettes = evaluate_cluster_counts(X, k_range)
        k_elbow = find_best_k_elbow(inertias, k_range)
        k_sil = find_best_k_silhouette(silhouettes, k_range)
        print(f"  Elbow suggests k={k_elbow}, Silhouette suggests k={k_sil}")
        n_clusters = k_sil  # Prefer silhouette for K-means

    print(f"  Fitting K-means with k={n_clusters}...")
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)

    score = silhouette_score(X, labels)
    print(f"  Silhouette score: {score:.4f}")

    df = df.copy()
    df["cluster"] = labels

    if objective == "A":
        prefix, out_prefix = "supplier", "philgeps_supplier"
    elif objective == "B":
        prefix, out_prefix = "procurement", "philgeps_procurement"
    else:
        prefix, out_prefix = "distribution", "doh_distribution"
    model_path = os.path.join(output_dir, f"kmeans_{prefix}_k{n_clusters}.pkl")
    labels_path = os.path.join(output_dir, f"{out_prefix}_clustered.csv")

    with open(model_path, "wb") as f:
        pickle.dump({"model": km, "n_clusters": n_clusters, "silhouette": score}, f)
    df.to_csv(labels_path, index=False)

    print(f"  Saved model: {model_path}")
    print(f"  Saved clustered data: {labels_path}")
    return km, labels, df, score


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("K-means Modeling - PhilGEPS & DOH Medical Data")
    print("=" * 60)

    # Objective A: Supplier Clustering
    print("\n[Objective A] Supplier / Awardee Clustering")
    run_kmeans_and_save("A", DEFAULT_K_A, use_suggested=True)

    # Objective B: Medicine Procurement Pattern
    print("\n[Objective B] Medicine Procurement Pattern Clustering")
    run_kmeans_and_save("B", DEFAULT_K_B, use_suggested=True)

    # Objective C: DOH Distribution Pattern
    print("\n[Objective C] Distribution Pattern Clustering (DOH)")
    run_kmeans_and_save("C", DEFAULT_K_C, use_suggested=True)

    print("\nModeling complete. Outputs in model_outputs/")


if __name__ == "__main__":
    main()
