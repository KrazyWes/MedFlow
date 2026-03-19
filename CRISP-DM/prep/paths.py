import os


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_visualization_paths(project_root: str) -> dict:
    """
    Current visualization folder layout (matches your latest renames):

    webp/
      logs/
        data_preparation_run.txt
      EDA/
        data_preparation/
          reports_txt/
            data_understanding_before.txt
            data_understanding_after.txt
          before/
            (EDA PNG charts for raw stage)
          after/
            (EDA PNG charts for cleaned/imputed stage)
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

