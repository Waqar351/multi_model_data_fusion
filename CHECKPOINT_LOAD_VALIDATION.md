# Checkpoint load-only validation

Validated on 17 August 2026 using the `sib-data-fusion` Poetry environment.
Every notebook loaded its standardized best-validation checkpoint and completed
its downstream test and explainability cells without executing its training loop.

| Notebook | Loaded epoch | Validation MSE | Result |
|---|---:|---:|---|
| M7_Late_Fusion_Complete_Explainability.ipynb | 148 | 0.0006502891 | PASS |
| M11_Gate_Scalar_Complete_Explainability.ipynb | 171 | 0.000080360313 | PASS |
| M11_Gate_Vector_only_Explainability.ipynb | 191 | 0.000083734015 | PASS |
| M33_Hierarchical_Fusion_Complete_Explainability.ipynb | 18 | 0.00017196472 | PASS |
| Original_GMU_Only_Complete_Explainability.ipynb | 124 | 0.000090534086 | PASS |
| GMU_GNN_Complete_Explainability.ipynb | 105 | 0.00010925254 | PASS |

The M33 checkpoint contains training/validation loss history but does not contain
the historical static/dynamic encoder-norm series. The diagnostic now preserves
the saved loss curves and computes static/dynamic encoder norm shares directly
from the loaded best model over the chronological validation windows. It writes
the normal `hierarchical_training_diagnostics.png` output without retraining.
All test-time and explainability analyses execute.

The validation runner uses a non-interactive plotting backend, so warnings that
figures cannot be shown are expected; the figures are still saved by the notebook
cells. Other observed warnings (pandas `applymap` deprecation and joblib CPU-count
detection) are non-fatal and unrelated to checkpoint loading.
