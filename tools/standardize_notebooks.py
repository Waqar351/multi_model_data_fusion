import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = {
    "M7_Late_Fusion_Complete_Explainability.ipynb": "M7_Late_Fusion_Complete_Explainability",
    "M11_Gate_Scalar_Complete_Explainability.ipynb": "M11_Gate_Scalar_Complete_Explainability",
    "M11_Gate_Vector_only_Explainability.ipynb": "M11_Gate_Vector_only_Explainability",
    "M33_Hierarchical_Fusion_Complete_Explainability.ipynb": "M33_Hierarchical_Fusion_Complete_Explainability",
    "GMU_GNN_Complete_Explainability.ipynb": "GMU_GNN_Complete_Explainability",
    "Original_GMU_Only_Complete_Explainability.ipynb": "Original_GMU_Only_Complete_Explainability",
}


STANDARD_CONFIG = '''

# Publication-comparison standardization (source update; rerun required).
model_used = {model_used!r}
results_root = os.path.join("results", model_used)
checkpoint_dir = os.path.join(results_root, "checkpoints")
plot_dir = os.path.join(results_root, "plots")
csv_dir = os.path.join(results_root, "csv")
for directory in (results_root, checkpoint_dir, plot_dir, csv_dir):
    os.makedirs(directory, exist_ok=True)

# Backward-compatible destination for existing analysis cells.
output_folder = results_root
execution_status = "SOURCE_UPDATED_REQUIRES_CLEAN_RERUN"
split_type = "chronological_transductive"
prediction_horizon = 1
validation_fraction = 0.20
max_epochs = 250
patience = 20
min_delta = 1e-7
batch_size = 1
comparison_seeds = [42, 43, 44, 45, 46]

def standardized_checkpoint(
    model, optimizer, best_epoch, best_validation_loss,
    train_loss_history, validation_loss_history,
    best_model_state=None, best_optimizer_state=None,
    diagnostic_histories=None,
):
    """Build the common, reloadable comparison checkpoint."""
    import torch_geometric
    model_state = best_model_state if best_model_state is not None else model.state_dict()
    optimizer_state_matches_best_epoch = best_optimizer_state is not None
    optimizer_state = (
        best_optimizer_state if best_optimizer_state is not None
        else optimizer.state_dict()
    )
    static_names = list(static_dt.columns) if "static_dt" in globals() else []
    dynamic_names = list(dynamic_cols) if "dynamic_cols" in globals() else []
    train_at_best = None
    if best_epoch and len(train_loss_history) >= best_epoch:
        train_at_best = float(train_loss_history[best_epoch - 1])
    return {{
        "schema_version": 1,
        "model_name": model_used,
        "architecture_name": type(model).__name__,
        "model_class": type(model).__name__,
        "model_config": {{
            "static_dim": int(static_tensor.shape[1]),
            "dynamic_dim": int(W),
            "window_size": int(W),
            "prediction_horizon": int(prediction_horizon),
        }},
        "model_state_dict": model_state,
        "optimizer_name": type(optimizer).__name__,
        "optimizer_config": {{
            "lr": float(optimizer.param_groups[0]["lr"]),
            "weight_decay": float(optimizer.param_groups[0].get("weight_decay", 0.0)),
        }},
        "optimizer_state_dict": optimizer_state,
        "optimizer_state_matches_best_epoch": optimizer_state_matches_best_epoch,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "train_loss_at_best_epoch": train_at_best,
        "train_loss_history": list(train_loss_history),
        "validation_loss_history": list(validation_loss_history),
        "diagnostic_histories": diagnostic_histories or {{}},
        "seed": int(seed),
        "dataset_path": str(name_file_dataset),
        "edge_path": "datasets/aristas_subgrafoSPdaily.csv",
        "static_feature_names": static_names,
        "dynamic_column_names": dynamic_names,
        "static_dim": int(static_tensor.shape[1]),
        "dynamic_dim": int(W),
        "window_size": int(W),
        "prediction_horizon": int(prediction_horizon),
        "preprocessing_config": {{
            "smoothing": "EWM",
            "ewm_alpha": 0.1,
            "transform": "log1p",
            "missing_values": "torch.nan_to_num(nan=0.0)",
            "static_scaler": "MinMaxScaler_fit_on_training_nodes",
            "dynamic_scaler": "global_minmax_fit_on_training_nodes_and_training_times",
        }},
        "split_config": {{
            "type": split_type,
            "train_end": int(TRAIN_END),
            "validation_fraction": float(validation_fraction),
            "fit_windows": len(fit_dataset),
            "validation_windows": len(validation_dataset),
            "test_windows": len(test_dataset),
        }},
        "train_node_ids": train_nodes.detach().cpu(),
        "test_node_ids": test_nodes.detach().cpu(),
        "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "pytorch_version": torch.__version__,
        "pyg_version": torch_geometric.__version__,
        "execution_status": execution_status,
    }}
'''


RERUN_NOTICE = '''## Reproducibility status

The source in this notebook has been standardized for the publication comparison. Existing saved outputs are retained as historical evidence, but they must not be combined with the updated source. Run the notebook from a clean kernel; final test metrics and explanations are valid only after the standardized best-validation checkpoint is reloaded.

The active protocol is chronological and transductive. The optional unseen-node wrapper is disabled. Case-level correlation p-values in retained historical outputs are descriptive because node-window rows are temporally clustered; publication inference must use the time-cluster bootstrap intervals and matched-seed variation. Frozen zero-input modality evaluations are ablations, not independently trained unimodal baselines.
'''


def source(cell):
    return "".join(cell.get("source", []))


def set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def replace_all_code(nb, old, new):
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            set_source(cell, source(cell).replace(old, new))


def append_once(text, addition, marker):
    return text if marker in text else text.rstrip() + "\n" + addition + "\n"


def find_code(nb, needle):
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code" and needle in source(cell):
            return cell
    raise RuntimeError(f"Could not find code cell containing {needle!r}")


def standardize_common(nb, model_used):
    # Keep PyG batching unambiguous even when wildcard imports expose torch DataLoader.
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            s = source(cell)
            if "from torch_geometric.loader import DataLoader as PyGDataLoader" not in s:
                s = s.replace(
                    "from torch_geometric.loader import DataLoader",
                    "from torch_geometric.loader import DataLoader as PyGDataLoader",
                )
            set_source(cell, s)
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            s = re.sub(r"(?<![A-Za-z_])DataLoader\(", "PyGDataLoader(", source(cell))
            s = s.replace("dynamic_cols = [col for col in df_nodes_label.columns if re.match", "dynamic_cols = sorted([col for col in df_nodes_label.columns if re.match")
            if "dynamic_cols = sorted([" in s and "col)]" in s:
                s = s.replace("col)]\n", "col)])\n", 1)
            set_source(cell, s)

    config = find_code(nb, "data_set =")
    s = source(config)
    # Remove legacy assignments that would override the mandated value/path.
    if "SOURCE_UPDATED_REQUIRES_CLEAN_RERUN" not in s:
        s = re.sub(r'^model_used\s*=.*$', '# legacy model_used replaced below.', s, flags=re.M)
        s = re.sub(r'^output_folder\s*=.*$', '# legacy output_folder replaced below.', s, flags=re.M)
        s = re.sub(r'^os\.makedirs\(output_folder, exist_ok=True\)\s*$', '', s, flags=re.M)
    s = append_once(s, STANDARD_CONFIG.format(model_used=model_used), "SOURCE_UPDATED_REQUIRES_CLEAN_RERUN")
    if "optimizer_state_matches_best_epoch = best_optimizer_state is not None" not in s:
        s = s.replace(
            "model_state = best_model_state if best_model_state is not None else model.state_dict()",
            "model_state = best_model_state if best_model_state is not None else model.state_dict()\n"
            "    optimizer_state_matches_best_epoch = best_optimizer_state is not None",
        )
    if '"optimizer_state_matches_best_epoch": optimizer_state_matches_best_epoch' not in s:
        s = s.replace(
            '"optimizer_state_dict": optimizer_state,',
            '"optimizer_state_dict": optimizer_state,\n'
            '        "optimizer_state_matches_best_epoch": optimizer_state_matches_best_epoch,',
        )
    if not re.search(r'^model_used\s*=', s, flags=re.M):
        s = s.replace(
            '# Publication-comparison standardization (source update; rerun required).',
            '# Publication-comparison standardization (source update; rerun required).\n'
            f'model_used = {model_used!r}',
        )
    if not re.search(r'^output_folder\s*=', s, flags=re.M):
        s = s.replace(
            '# Backward-compatible destination for existing analysis cells.',
            '# Backward-compatible destination for existing analysis cells.\n'
            'output_folder = results_root',
        )
    if "comparison_seeds =" not in s:
        s = s.replace("batch_size = 1", "batch_size = 1\ncomparison_seeds = [42, 43, 44, 45, 46]")
    set_source(config, s)

    # Remove later legacy overrides, while leaving the standardized config authoritative.
    for cell in nb["cells"]:
        if cell is config or cell.get("cell_type") != "code":
            continue
        s = re.sub(r'^model_used\s*=.*$', '# legacy model_used override removed', source(cell), flags=re.M)
        s = re.sub(r'^output_folder\s*=.*$', '# legacy output_folder override removed', s, flags=re.M)
        set_source(cell, s)

    # Make the actual active protocol unambiguous.
    for cell in nb["cells"]:
        if cell.get("cell_type") == "markdown":
            s = source(cell)
            s = re.sub(r"# Inductive Learning", "# Optional unseen-node inductive workflow (disabled)", s, flags=re.I)
            s = re.sub(r"# Inductive learning split", "# Optional unseen-node inductive workflow (disabled)", s, flags=re.I)
            s = re.sub(r"# Inductive Split on nodes", "# Active chronological transductive split", s, flags=re.I)
            set_source(cell, s)

    wrapper = find_code(nb, "def filter_and_reindex")
    s = source(wrapper)
    if "new_data.original_node_ids" not in s:
        s = s.replace(
            "new_data.node_ids = new_node_ids",
            "new_data.original_node_ids = old_node_ids.clone()  # preserve global identity\n    new_data.node_ids = new_node_ids",
        )
    set_source(wrapper, s)

    # Explicitly document that the wrapper is inactive in the current experiment.
    protocol_note = (
        '\n# Active protocol: chronological and transductive. The optional\n'
        '# InductiveWrapperDataset is intentionally not applied here.\n'
        'split_type = "chronological_transductive"\n'
    )
    # Remove runtime assertions from any prior migration pass. Protocol labeling
    # must not depend on train_dataset already existing when a cell is executed.
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = re.sub(
            r'\n# Active protocol: all nodes are present in every chronological window\.\n'
            r'split_type = "chronological_transductive"\n'
            r'(?:assert "train_dataset" in globals\(\) and "test_dataset" in globals\(\)\n)?'
            r'assert not isinstance\(train_dataset, InductiveWrapperDataset\)\n'
            r'assert not isinstance\(test_dataset, InductiveWrapperDataset\)\n?',
            '\n', source(cell),
        )
        s = re.sub(
            r'\n# Active protocol: chronological and transductive\. The optional\n'
            r'# InductiveWrapperDataset is intentionally not applied here\.\n'
            r'split_type = "chronological_transductive"\n?',
            '\n', s,
        )
        set_source(cell, s)

    # Select the active cell using an un-commented assignment and place the audit
    # only after both train_dataset and test_dataset have been constructed.
    active_dataset_cells = [
        cell for cell in nb["cells"]
        if cell.get("cell_type") == "code"
        and re.search(r'^train_dataset\s*=\s*CrimeWindowPyGDataset\(', source(cell), re.M)
        and re.search(r'^test_dataset\s*=\s*CrimeWindowPyGDataset\(', source(cell), re.M)
    ]
    if len(active_dataset_cells) != 1:
        raise RuntimeError(f"Expected one active dataset-construction cell, found {len(active_dataset_cells)}")
    dataset_cell = active_dataset_cells[0]
    set_source(dataset_cell, source(dataset_cell).rstrip() + protocol_note)

    # Add a visible source/output provenance notice without deleting historical output.
    existing_notice = next((c for c in nb["cells"] if "## Reproducibility status" in source(c)), None)
    if existing_notice is not None:
        set_source(existing_notice, RERUN_NOTICE)
    else:
        nb["cells"].insert(1, {
            "cell_type": "markdown",
            "metadata": {},
            "source": RERUN_NOTICE.splitlines(keepends=True),
        })


def add_checkpoint_save(cell, history_train, history_val, model_state, optimizer_state, diagnostics="{}"):
    s = source(cell)
    addition = f'''

# Save the common comparison checkpoint after restoring the selected model.
checkpoint_path = os.path.join(checkpoint_dir, f"{{model_used}}__seed-{{seed}}__best-val.pt")
checkpoint = standardized_checkpoint(
    model=model,
    optimizer=optimizer,
    best_epoch=best_epoch,
    best_validation_loss=best_validation_loss,
    train_loss_history={history_train},
    validation_loss_history={history_val},
    best_model_state={model_state},
    best_optimizer_state=globals().get({optimizer_state!r}),
    diagnostic_histories={diagnostics},
)
torch.save(checkpoint, checkpoint_path)

# Reload from disk so every downstream result comes from the serialized best state.
checkpoint_device = next(model.parameters()).device
checkpoint = torch.load(checkpoint_path, map_location=checkpoint_device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.eval()
assert checkpoint["best_epoch"] == best_epoch
execution_status = "EXECUTED_FROM_STANDARDIZED_BEST_CHECKPOINT"
print("Restored standardized best-validation checkpoint:", checkpoint_path)
'''
    s = append_once(s, addition, "Restored standardized best-validation checkpoint:")
    # Upgrade checkpoint calls created by older migration versions so a detached
    # save cell cannot fail merely because the optional best optimizer snapshot
    # is absent from the live kernel. The schema records whether it truly matches.
    s = re.sub(
        r'best_optimizer_state=best_optimizer_state,',
        'best_optimizer_state=globals().get("best_optimizer_state"),',
        s,
    )
    legacy = "checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)"
    if "# Reload from disk so every downstream result" in s and legacy in s:
        head, tail = s.rsplit(legacy, 1)
        s = head + (
            "checkpoint_device = next(model.parameters()).device\n"
            "checkpoint = torch.load(checkpoint_path, map_location=checkpoint_device, weights_only=False)"
        ) + tail
    if "Restored standardized best-validation checkpoint:" in s and "EXECUTED_FROM_STANDARDIZED_BEST_CHECKPOINT" not in s:
        s = s.replace(
            'print("Restored standardized best-validation checkpoint:", checkpoint_path)',
            'execution_status = "EXECUTED_FROM_STANDARDIZED_BEST_CHECKPOINT"\n'
            'print("Restored standardized best-validation checkpoint:", checkpoint_path)',
        )
    set_source(cell, s)


def standardize_training(nb, name):
    # Harmonize comparison budget without altering architecture.
    replace_all_code(nb, "patience = 15", "patience = 20")
    replace_all_code(nb, "min_delta = 1e-6", "min_delta = 1e-7")

    if name.startswith("M7_"):
        replace_all_code(nb, "num_epochs = 200", "num_epochs = 250")
        cell = find_code(nb, "Restored epoch")
        # M7 previously had no optimizer snapshot.
        s = source(cell)
        s = re.sub(
            r'best_state = None(?:\nbest_optimizer_state = None)+',
            'best_state = None\nbest_optimizer_state = None', s,
        )
        if "best_optimizer_state = None" not in s:
            s = s.replace("best_state = None", "best_state = None\nbest_optimizer_state = None")
        s = re.sub(
            r'best_state = copy\.deepcopy\(model\.state_dict\(\)\)'
            r'(?:\n        best_optimizer_state = copy\.deepcopy\(optimizer\.state_dict\(\)\))+',
            'best_state = copy.deepcopy(model.state_dict())\n'
            '        best_optimizer_state = copy.deepcopy(optimizer.state_dict())', s,
        )
        if "best_optimizer_state = copy.deepcopy(optimizer.state_dict())" not in s:
            s = s.replace(
                "best_state = copy.deepcopy(model.state_dict())",
                "best_state = copy.deepcopy(model.state_dict())\n"
                "        best_optimizer_state = copy.deepcopy(optimizer.state_dict())",
            )
        set_source(cell, s)
        add_checkpoint_save(
            cell, "train_loss_per_epoch", "validation_loss_per_epoch",
            "best_state", "best_optimizer_state",
            '{"modality_norm_history": norm_hist, "conditional_norm_history": cond_norm_hist, "gradient_history": grad_hist}',
        )
    elif name.startswith("M33_"):
        cell = find_code(nb, "No validation checkpoint was created")
        s = source(cell)
        if "best_optimizer_state" not in s:
            s = s.replace("best_model_state = None", "best_model_state = None\nbest_optimizer_state = None")
            s = s.replace(
                "best_model_state = copy.deepcopy(model.state_dict())",
                "best_model_state = copy.deepcopy(model.state_dict())\n        best_optimizer_state = copy.deepcopy(optimizer.state_dict())",
            )
        set_source(cell, s)
        add_checkpoint_save(cell, "train_loss_history", "validation_loss_history", "best_model_state", "best_optimizer_state")
    elif name.startswith("M11_Gate_Vector"):
        cell = find_code(nb, "No valid checkpoint was created")
        s = source(cell)
        if not re.search(r'^best_optimizer_state\s*=\s*None', s, re.M):
            s = s.replace("best_model_state = None", "best_model_state = None\n\nbest_optimizer_state = None")
        if "best_optimizer_state = copy.deepcopy" not in s:
            s = s.replace(
                "best_model_state = copy.deepcopy(\n            model.state_dict()\n        )",
                "best_model_state = copy.deepcopy(\n            model.state_dict()\n        )\n\n"
                "        best_optimizer_state = copy.deepcopy(\n"
                "            optimizer.state_dict()\n"
                "        )",
            )
        s = s.replace(
            'diagnostic_histories={"self_gate_history": self_gate_history, "neighbor_gate_history": neighbor_gate_history}',
            'diagnostic_histories={"self_static_gate_history": self_static_gate_history, '
            '"neighbor_static_gate_history": neighbor_static_gate_history}',
        )
        set_source(cell, s)
        add_checkpoint_save(
            cell, "train_loss_history", "validation_loss_history",
            "best_model_state", "best_optimizer_state",
            '{"self_static_gate_history": self_static_gate_history, "neighbor_static_gate_history": neighbor_static_gate_history}',
        )
    elif name.startswith("M11_Gate_Scalar"):
        cell = find_code(nb, "max_epochs=250")
        # Existing cell already saves and reloads; add standardized reload afterward.
        add_checkpoint_save(
            cell, "train_loss_history", "validation_loss_history",
            "best_model_state", "best_optimizer_state",
            '{"alpha_self_history": alpha_self_history, "alpha_neigh_history": alpha_neigh_history}',
        )
    elif name in ("GMU_GNN_Complete_Explainability.ipynb", "Original_GMU_Only_Complete_Explainability.ipynb"):
        marker = "best_gmu_gnn_chronological.pt" if name.startswith("GMU_GNN") else "best_original_gmu_chronological.pt"
        cell = find_code(nb, marker)
        # The old code saved best states but failed to restore them.
        add_checkpoint_save(
            cell, "train_loss_history", "validation_loss_history",
            "best_model_state", "best_optimizer_state",
            '{"static_gate_history": static_gate_history, "validation_static_gate_history": validation_static_gate_history}',
        )


def add_common_final_cells(nb, model_used):
    no_graph = model_used.startswith("Original_GMU")
    summary_code = f'''# Standard final comparison record (self-contained checkpoint reload).
checkpoint_path = os.path.join(
    checkpoint_dir, f"{{model_used}}__seed-{{seed}}__best-val.pt"
)
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"Standardized checkpoint not found: {{checkpoint_path}}. "
        "Run the training cell first."
    )

checkpoint_device = next(model.parameters()).device
checkpoint = torch.load(
    checkpoint_path, map_location=checkpoint_device, weights_only=False
)
model.load_state_dict(checkpoint["model_state_dict"])
if "optimizer_state_dict" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.eval()
execution_status = "EXECUTED_FROM_STANDARDIZED_BEST_CHECKPOINT"

metric_candidates = [
    globals().get("metrics"),
    globals().get("scalar_metrics"),
    globals().get("hierarchical_metrics"),
    globals().get("test_metrics"),
]
final_metrics = next((m for m in metric_candidates if isinstance(m, dict)), None)
if final_metrics is None:
    raise RuntimeError("No final metric dictionary is available; run final test evaluation first.")

def _metric(name):
    for key, value in final_metrics.items():
        if str(key).lower().replace("²", "2") == name:
            return float(value)
    return np.nan

publication_comparison_summary = pd.DataFrame([{{
    "model": model_used,
    "seed": seed,
    "split": split_type,
    "n_windows": len(test_dataset),
    "n_nodes": int(static_tensor.shape[0]),
    "n_cases": int(len(test_dataset) * static_tensor.shape[0]),
    "mse": _metric("mse"),
    "rmse": _metric("rmse"),
    "mae": _metric("mae"),
    "r2": _metric("r2"),
    "huber": _metric("huber"),
    "uses_graph": {str(not no_graph)},
    "execution_status": execution_status,
}}])

publication_comparison_summary.to_csv(
    os.path.join(csv_dir, f"{{model_used}}__seed-{{seed}}__summary.csv"), index=False
)
display(publication_comparison_summary)
'''
    existing = next((
        c for c in nb["cells"]
        if c.get("cell_type") == "code" and "publication_comparison_summary" in source(c)
    ), None)
    if existing is not None:
        set_source(existing, summary_code)
        return
    nb["cells"].extend([
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ("## Standard publication-comparison export\n\n"
                       "This export is guarded: it runs only after a clean best-checkpoint reload and a real final metric dictionary. Architecture-inapplicable fields must remain `NaN`, not fabricated.\n").splitlines(keepends=True),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": summary_code.splitlines(keepends=True),
        },
    ])


def main():
    for filename, model_used in NOTEBOOKS.items():
        path = ROOT / filename
        nb = json.loads(path.read_text(encoding="utf-8"))
        standardize_common(nb, model_used)
        standardize_training(nb, filename)
        add_common_final_cells(nb, model_used)
        # Remove retained exception tracebacks from previously failed cells. Valid
        # historical outputs remain untouched; failed cells are marked unexecuted.
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            outputs = cell.get("outputs", [])
            if any(output.get("output_type") == "error" for output in outputs):
                cell["outputs"] = [
                    output for output in outputs
                    if output.get("output_type") != "error"
                ]
                cell["execution_count"] = None
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"standardized {filename}")


if __name__ == "__main__":
    main()
