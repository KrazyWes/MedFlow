import os
import pandas as pd


CSV_EXTENSIONS = (".csv",)
EXCEL_EXTENSIONS = (".xlsx", ".xls")
ALL_DATA_EXTENSIONS = CSV_EXTENSIONS + EXCEL_EXTENSIONS


def get_supported_files(directory: str):
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
    """Load a CSV file. Uses low_memory=False to avoid mixed-type warnings."""
    return pd.read_csv(path, low_memory=False)


def load_excel(path: str, dtype=str) -> list[pd.DataFrame]:
    """Load an Excel file. Returns list of DataFrames (one per sheet)."""
    xl = pd.ExcelFile(path)
    kwargs = {"dtype": dtype} if dtype is not None else {}
    return [pd.read_excel(path, sheet_name=name, **kwargs) for name in xl.sheet_names]

