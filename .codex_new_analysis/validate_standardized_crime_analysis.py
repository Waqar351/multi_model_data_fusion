import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"D:\data_fusion")
SPECS = [
    ("M7_Late_Fusion_Complete_Explainability.ipynb", "M7_Late_Fusion_Complete_Explainability", "late_fusion_effective_contribution_vs_crime.png"),
    ("M11_Gate_Scalar_Complete_Explainability.ipynb", "M11_Gate_Scalar_Complete_Explainability", "scalar_effective_activation_contribution_vs_crime.png"),
    ("M33_Hierarchical_Fusion_Complete_Explainability.ipynb", "M33_Hierarchical_Fusion_Complete_Explainability", "hierarchical_effective_contribution_vs_crime.png"),
]

for notebook_name, result_dir, expected_png in SPECS:
    notebook = json.loads((ROOT / notebook_name).read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][-1]["source"])
    output_folder = str(ROOT / "results_new" / result_dir)
    namespace = {"os": os, "np": np, "pd": pd, "plt": plt, "output_folder": output_folder}
    exec(compile(code, notebook_name, "exec"), namespace)
    plt.close("all")
    output = Path(output_folder) / expected_png
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"Expected output was not generated: {output}")
    print(f"Validated {notebook_name}: {output.name}")
