import os
import pandas as pd

# Paths: input from raw_datasets, output to this_datasets
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
input_dir = os.path.join(project_root, "raw_datasets")
output_dir = os.path.join(project_root, "this_datasets")

input_files = [
    "2025_(JAN-MAR).csv",
    "2025_(APR-JUN).csv",
    "2025_(JUL-SEPT).csv",
]

dfs = []
for f in input_files:
    path = os.path.join(input_dir, f)
    if os.path.exists(path):
        dfs.append(pd.read_csv(path))
        print(f"Loaded: {f}")
    else:
        print(f"Skipped (not found): {f}")

if not dfs:
    raise FileNotFoundError(f"None of the input files found: {input_files}")

df = pd.concat(dfs, ignore_index=True)

# List of medical-related keywords
medical_keywords = [
    "medical", "medicine", "pharmaceutical", "drug", "vaccine",
    "hospital", "laboratory", "diagnostic", "surgical",
    "clinic", "health", "therapeutic", "antibiotic",
    "syringe", "test kit", "reagent", "biomedical"
]

# Create regex pattern from keywords
pattern = "|".join(medical_keywords)

# Filter rows where UNSPSC Description contains medical keywords
medical_df = df[df["UNSPSC Description"].str.contains(pattern, case=False, na=False)]

# Save filtered dataset
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "medical_related_procurement.csv")
medical_df.to_csv(output_file, index=False)

print("Filtering complete.")
print("Total medical-related records:", len(medical_df))
print("Saved to:", output_file)