"""
Sequential runner for all pipeline steps.

Order matters: step 02 expects CSVs from step 01, step 04 expects `*_features_minmax.csv` from 02, and so on.
K-Means (04) is listed before DBSCAN (04) so both algorithms see the same upstream artifacts;
steps 05-07 then run k-means then DBSCAN for each analysis step.

Working directory: this file's directory (`CRISP-DM/`). Step scripts add the parent folder to
`sys.path` so they can `import _common` and `sources_paths`.

Outputs land under `this_datasets/{DOH|PhilGEPS}/`, `webp/EDA_and_visualization/{DOH|PhilGEPS}/`,
and `webp/logs/{DOH|PhilGEPS}/` (see `OUTPUT_LAYOUT.md`).

To clear those trees first: `python 00_clear_pipeline_outputs.py` (or `python main.py --fresh`).
"""
import os
import subprocess
import sys

# Relative paths; executed with cwd = directory containing this file.
SCRIPTS = [
    # --- DOH ---
    os.path.join("DOH", "01_data_cleaning_doh.py"),
    os.path.join("DOH", "02_data_transformation_doh.py"),
    os.path.join("DOH", "03_exploratory_data_analysis_doh.py"),
    os.path.join("DOH", "04_clustering_implementation_kmeans_doh.py"),
    os.path.join("DOH", "04_clustering_implementation_dbscan_doh.py"),
    os.path.join("DOH", "05_cluster_analysis_kmeans_doh.py"),
    os.path.join("DOH", "05_cluster_analysis_dbscan_doh.py"),
    os.path.join("DOH", "06_visualization_kmeans_doh.py"),
    os.path.join("DOH", "06_visualization_dbscan_doh.py"),
    os.path.join("DOH", "07_evaluation_kmeans_doh.py"),
    os.path.join("DOH", "07_evaluation_dbscan_doh.py"),
    os.path.join("DOH", "08_final_output_bundle_doh.py"),
    # --- PhilGEPS ---
    os.path.join("PhilGEPS", "01_data_cleaning_philgeps.py"),
    os.path.join("PhilGEPS", "02_data_transformation_philgeps.py"),
    os.path.join("PhilGEPS", "03_exploratory_data_analysis_philgeps.py"),
    os.path.join("PhilGEPS", "04_clustering_implementation_kmeans_philgeps.py"),
    os.path.join("PhilGEPS", "04_clustering_implementation_dbscan_philgeps.py"),
    os.path.join("PhilGEPS", "05_cluster_analysis_kmeans_philgeps.py"),
    os.path.join("PhilGEPS", "05_cluster_analysis_dbscan_philgeps.py"),
    os.path.join("PhilGEPS", "06_visualization_kmeans_philgeps.py"),
    os.path.join("PhilGEPS", "06_visualization_dbscan_philgeps.py"),
    os.path.join("PhilGEPS", "07_evaluation_kmeans_philgeps.py"),
    os.path.join("PhilGEPS", "07_evaluation_dbscan_philgeps.py"),
    os.path.join("PhilGEPS", "08_final_output_bundle_philgeps.py"),
]

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for rel in SCRIPTS:
        path = os.path.join(script_dir, rel)
        if not os.path.exists(path):
            print(f"Skip (not found): {rel}")
            continue
        print(f"\n{'='*60}\nRunning: {rel}\n{'='*60}")
        # cwd=script_dir keeps relative paths inside each step script consistent.
        code = subprocess.call([sys.executable, path], cwd=script_dir)
        if code != 0:
            print(f"FAILED: {rel} (exit code {code})")
            sys.exit(code)
    print(f"\n{'='*60}\nAll pipeline steps completed successfully.\n{'='*60}")
