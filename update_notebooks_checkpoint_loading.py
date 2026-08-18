"""Add load-first checkpoint guards to the six explainability notebooks."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOKS = {
    'M7_Late_Fusion_Complete_Explainability.ipynb': {'training': 54, 'extra': []},
    'M11_Gate_Scalar_Complete_Explainability.ipynb': {'training': 57, 'extra': []},
    'M11_Gate_Vector_only_Explainability.ipynb': {'training': 52, 'extra': [53]},
    'M33_Hierarchical_Fusion_Complete_Explainability.ipynb': {'training': 51, 'extra': []},
    'Original_GMU_Only_Complete_Explainability.ipynb': {'training': 51, 'extra': []},
    'GMU_GNN_Complete_Explainability.ipynb': {'training': 53, 'extra': [54]},
}

MARKER = '# LOAD-FIRST CHECKPOINT GUARD (added for analysis-only reruns)'

PREFIX = '''# LOAD-FIRST CHECKPOINT GUARD (added for analysis-only reruns)
# The standardized best-validation checkpoint is authoritative.  Training runs
# only when it is absent and ALLOW_TRAINING_IF_CHECKPOINT_MISSING is True.
ALLOW_TRAINING_IF_CHECKPOINT_MISSING = True
criterion = nn.MSELoss()
checkpoint_path = os.path.join(
    checkpoint_dir, f"{model_used}__seed-{seed}__best-val.pt"
)
checkpoint_loaded = False

if os.path.isfile(checkpoint_path):
    checkpoint_device = next(model.parameters()).device
    checkpoint = torch.load(
        checkpoint_path, map_location=checkpoint_device, weights_only=False
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint has no model_state_dict: {checkpoint_path}")
    saved_model_name = checkpoint.get("model_name")
    if saved_model_name not in (None, model_used):
        raise ValueError(
            f"Checkpoint model mismatch: expected {model_used!r}, "
            f"found {saved_model_name!r}."
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (ValueError, KeyError) as exc:
            print(f"Optimizer state not restored (analysis is unaffected): {exc}")
    model.eval()

    best_epoch = int(checkpoint.get("best_epoch", 0) or 0)
    best_validation_loss = float(
        checkpoint.get("best_validation_loss", float("nan"))
    )
    train_loss_history = list(checkpoint.get("train_loss_history", []))
    validation_loss_history = list(
        checkpoint.get("validation_loss_history", [])
    )
    # Preserve aliases used by the different notebooks' downstream plots.
    train_loss_per_epoch = train_loss_history
    validation_loss_per_epoch = validation_loss_history
    test_loss_per_epoch = validation_loss_history
    diagnostic_histories = checkpoint.get("diagnostic_histories", {}) or {}
    norm_hist = list(diagnostic_histories.get("modality_norm_history", []))
    cond_norm_hist = list(diagnostic_histories.get("conditional_norm_history", []))
    grad_hist = list(diagnostic_histories.get("gradient_history", []))
    static_encoder_norm_history = list(
        diagnostic_histories.get("static_encoder_norm_history", [])
    )
    dynamic_encoder_norm_history = list(
        diagnostic_histories.get("dynamic_encoder_norm_history", [])
    )
    alpha_self_history = list(diagnostic_histories.get("alpha_self_history", []))
    alpha_neigh_history = list(diagnostic_histories.get("alpha_neigh_history", []))
    self_static_gate_history = list(
        diagnostic_histories.get("self_static_gate_history", [])
    )
    neighbor_static_gate_history = list(
        diagnostic_histories.get("neighbor_static_gate_history", [])
    )
    static_gate_history = list(diagnostic_histories.get("static_gate_history", []))
    validation_static_gate_history = list(
        diagnostic_histories.get("validation_static_gate_history", [])
    )
    best_train_loss = checkpoint.get("train_loss_at_best_epoch", float("nan"))
    num_trained_epochs = len(train_loss_history)
    validation_metrics_per_epoch = [{"MSE": value} for value in validation_loss_history]
    best_state = checkpoint["model_state_dict"]
    best_model_state = checkpoint["model_state_dict"]
    best_optimizer_state = checkpoint.get("optimizer_state_dict")
    checkpoint_loaded = True
    execution_status = "LOADED_EXISTING_BEST_CHECKPOINT_NO_TRAINING"
    print(f"Loaded existing checkpoint: {checkpoint_path}")
    print(
        f"Training skipped. Best epoch: {best_epoch}; "
        f"validation MSE: {best_validation_loss:.8g}"
    )
elif not ALLOW_TRAINING_IF_CHECKPOINT_MISSING:
    raise FileNotFoundError(
        f"Checkpoint not found and training is disabled: {checkpoint_path}"
    )
else:
    print(f"No checkpoint found; training will run: {checkpoint_path}")

if not checkpoint_loaded:
'''


def indent(source: str) -> str:
    return ''.join(('    ' + line if line.strip() else line) for line in source.splitlines(True))


def source_lines(source: str) -> list[str]:
    return source.splitlines(True)


def main() -> None:
    for filename, spec in NOTEBOOKS.items():
        path = Path(filename)
        nb = json.loads(path.read_text(encoding='utf-8'))
        cell = nb['cells'][spec['training']]
        original = ''.join(cell['source'])
        if MARKER in original:
            print(f'Skipped already-updated {filename}')
            continue
        cell['source'] = source_lines(PREFIX + indent(original))

        # These cells save/reload legacy checkpoints immediately after the main
        # training cell.  They must also be skipped on the load-first path.
        for index in spec['extra']:
            extra = nb['cells'][index]
            extra_source = ''.join(extra['source'])
            extra['source'] = source_lines(
                '# Skip legacy post-training checkpoint work when the standardized '
                'checkpoint was loaded.\nif not checkpoint_loaded:\n' + indent(extra_source)
                + ('\n' if not extra_source.endswith('\n') else '')
                + 'else:\n    print("Legacy checkpoint step skipped; standardized checkpoint is active.")\n'
            )

        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(f'Updated {filename}')


if __name__ == '__main__':
    main()
