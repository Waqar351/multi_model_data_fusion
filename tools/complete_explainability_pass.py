"""Apply the approved Phase 3 documentation and analysis completion pass."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "M7_Late_Fusion_Complete_Explainability.ipynb",
    "M11_Gate_Scalar_Complete_Explainability.ipynb",
    "M11_Gate_Vector_only_Explainability.ipynb",
    "M33_Hierarchical_Fusion_Complete_Explainability.ipynb",
    "Original_GMU_Only_Complete_Explainability.ipynb",
    "GMU_GNN_Complete_Explainability.ipynb",
]

STYLE_CELL = '''# Shared publication plotting configuration.
from explainability_plot_style import (
    MODEL_COLORS, STATIC_COLOR, DYNAMIC_COLOR, OBSERVED_COLOR,
    PREDICTION_COLOR, RESIDUAL_COLOR, REFERENCE_COLOR,
    FIGSIZE_SINGLE, FIGSIZE_DOUBLE, FIGSIZE_TRIPLE,
    TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE,
    LINE_WIDTH, MARKER_SIZE, DPI,
    apply_presentation_style, save_presentation_figure,
)
apply_presentation_style()
'''


def src(cell):
    return "".join(cell.get("source", []))


def set_src(cell, value):
    cell["source"] = value.splitlines(keepends=True)


def replace_strings(value, replacements):
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: replace_strings(item, replacements) for key, item in value.items()}
    return value


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def interpretation_for(title, model):
    low = title.lower()
    if "prediction" in low or "residual" in low:
        return (
            "The observed-versus-predicted panel evaluates calibration against the identity line. "
            "The residual panel uses prediction on the x-axis and observed minus predicted on the y-axis. "
            "Systematic curvature or changing spread indicates remaining error structure; it does not identify its cause."
        )
    if "gate health" in low or "gate distribution" in low:
        return (
            "Gate values are learned mixing coefficients. Distributions, entropy and saturation indicate whether the gate varies or collapses, "
            "but they do not by themselves measure contribution or establish that gating improves accuracy."
        )
    if "effective" in low or "contribution" in low or "branch strength" in low:
        if "M33" in model:
            return (
                "Static and dynamic quantities are occlusion-based prediction changes. Their normalized branch share is a diagnostic scale, "
                "not a probability gate; the non-additive interaction must be reported separately."
            )
        if "M7" in model:
            return (
                "The axes report absolute weighted branch outputs before the final bias. The normalized proportions summarize relative branch-output magnitude, "
                "not a learned gate and not a causal attribution."
            )
        return (
            "Effective gated activation combines learned coefficients with activation magnitude. The proportions indicate what enters fusion on this mechanistic scale; "
            "they should be interpreted alongside interventions and raw gate diagnostics."
        )
    if "specialization" in low or "channel" in low:
        return (
            "The plot compares latent dimensions or channels. Larger values identify dimensions carrying stronger static or dynamic signals, "
            "but latent dimensions do not necessarily correspond one-to-one with original input variables."
        )
    if "crime" in low or "association" in low:
        return (
            "The x-axis summarizes crime intensity and the y-axis summarizes gate or branch balance. Colors and legends distinguish static and dynamic roles. "
            "Associations are descriptive; temporal clustering and correlated inputs prevent causal interpretation."
        )
    if "stability" in low:
        return (
            "Node summaries describe spatial heterogeneity and time summaries describe temporal variation. Stable citywide means can coexist with meaningful node-level differences. "
            "These views assess robustness, not prediction quality by themselves."
        )
    if "error" in low:
        return (
            "The y-axis is absolute prediction error and the x-axis is the relevant gate or branch-balance statistic. "
            "A trend indicates association with difficult cases; it does not show that changing the gate would cause the error change."
        )
    if "intervention" in low or "ablation" in low or "removal" in low:
        return (
            "Bars report performance change under frozen-model input, branch, gate or graph manipulations. Larger degradation supports behavioral reliance, "
            "but these conditions are not independently trained unimodal baselines."
        )
    if "permutation" in low:
        return (
            "Feature importance is the increase in test MSE after permutation. Larger positive values indicate stronger predictive reliance under this perturbation. "
            "Correlated features can share or mask importance."
        )
    if "graph" in low or "neighbor" in low or "hop" in low:
        return (
            "Graph interventions compare the learned topology with removed or bypassed neighborhood information. The result measures reliance on graph structure in this trained model, "
            "not the universal value of spatial context."
        )
    if "bootstrap" in low or "confidence" in low:
        return (
            "Intervals use time-cluster resampling so node-window rows from the same period are not treated as independent. "
            "They quantify test-window variability and do not replace matched-seed retraining."
        )
    if "embedding" in low or "projection" in low:
        return (
            "Axes are nonlinear projection coordinates without direct physical meaning. Colors identify the encoded quantity or class. "
            "Separation is qualitative representation evidence and must not replace predictive or intervention metrics."
        )
    return (
        "Read axes and legends as defined in the generated figure. Compare this result only with architectures exposing an equivalent quantity, "
        "and avoid causal language unless the analysis is an explicit intervention."
    )


def add_interpretations(nb, model):
    for cell in nb["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        text = src(cell)
        if "### Presentation interpretation" in text:
            continue
        first = next((line.strip() for line in text.splitlines() if line.strip().startswith("## ")), None)
        if not first:
            continue
        title = first[3:].strip()
        if title.lower() in {"architecture", "what remains unchanged", "split nodes", "interpretation framework"}:
            continue
        if any(k in title.lower() for k in [
            "evaluation", "gate", "activation", "contribution", "specialization", "crime",
            "stability", "error", "intervention", "ablation", "permutation", "graph",
            "prediction", "embedding", "bootstrap", "neighborhood", "context", "role separation",
        ]):
            addition = f"\n\n### Presentation interpretation\n\n{interpretation_for(title, model)}\n"
            set_src(cell, text.rstrip() + addition)


def add_traceability(nb, model):
    if any("## Presentation traceability" in src(c) for c in nb["cells"]):
        return
    nb["cells"].append(md(f'''## Presentation traceability

This notebook is the source of truth for presentation evidence from **{model}**. Every slide figure must be generated by an executed code cell in this notebook and saved below `results/{model}`. Retained historical images must not be mixed with outputs from a different source revision.

### Canonical terminology

- **Gate coefficient** means a learned scalar or vector mixing value.
- **Effective gated activation proportion** means normalized magnitude after applying a gate to an activation.
- **Effective branch-output proportion** is used for ungated late fusion.
- **Occlusion-based branch share** is used for nonlinear hierarchical fusion.
- **Residual** is observed minus predicted; **absolute error** is its absolute value.
- Frozen-model interventions are not independently trained unimodal baselines.

### Validation requirement

Run from a clean kernel, confirm `execution_status == "LOADED_EXISTING_BEST_CHECKPOINT_NO_TRAINING"`, inspect every saved figure, and use only the regenerated files in the final presentation.
'''))


def complete_m7(nb):
    if not any("late_fusion_prediction_diagnostics.png" in src(c) for c in nb["cells"]):
        anchor = next(i for i,c in enumerate(nb["cells"]) if "late_frame.to_csv" in src(c))
        nb["cells"][anchor+1:anchor+1] = [
            md('''## B. Prediction diagnostics

This analysis compares test observations with predictions and inspects residual structure using the same definitions as the other five notebooks. It is applicable because the late-fusion collector retains one prediction and target for every evaluated node-time case.

### Presentation interpretation

The left panel plots observed crime against predicted crime with an identity reference. The right panel plots prediction against residual, defined as observed minus predicted. Tight identity alignment and residuals centered around zero indicate better calibration; visible structure is diagnostic and does not identify a causal mechanism.
'''),
            code('''y_true = late_frame["target_crime"].to_numpy()
y_pred = late_frame["prediction"].to_numpy()
residual = y_true - y_pred

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
axes[0].scatter(y_true, y_pred, s=14, alpha=0.32, color=PREDICTION_COLOR)
lo = min(y_true.min(), y_pred.min())
hi = max(y_true.max(), y_pred.max())
axes[0].plot([lo, hi], [lo, hi], color=REFERENCE_COLOR, linestyle="--", label="Identity")
axes[0].set(title="Observed vs predicted", xlabel="Observed crime", ylabel="Predicted crime")
axes[0].legend()

axes[1].scatter(y_pred, residual, s=14, alpha=0.32, color=RESIDUAL_COLOR)
axes[1].axhline(0, color=REFERENCE_COLOR, linestyle="--")
axes[1].set(title="Residual diagnostic", xlabel="Predicted crime", ylabel="Residual (observed - predicted)")

fig.tight_layout()
save_presentation_figure(fig, os.path.join(output_folder, "late_fusion_prediction_diagnostics.png"))
plt.show()

late_prediction_diagnostics = pd.DataFrame([{
    "pearson_observed_predicted": float(np.corrcoef(y_true, y_pred)[0, 1]),
    "residual_mean": float(np.mean(residual)),
    "residual_std": float(np.std(residual, ddof=1)),
    "residual_mae": float(np.mean(np.abs(residual))),
}])
display(late_prediction_diagnostics.round(6))
late_prediction_diagnostics.to_csv(
    os.path.join(output_folder, "late_fusion_prediction_diagnostics.csv"), index=False
)
''')]
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            s=src(c).replace('"Late-fusion effective static share"','"Late-fusion static branch-output proportion"')
            s=s.replace('"Static absolute-output share"','"Static branch-output proportion"')
            s=s.replace('"Effective branch share vs input crime"','"Branch-output proportion vs input crime"')
            s=s.replace('"Absolute-output share"','"Branch-output proportion"')
            set_src(c,s)


def complete_m33(nb):
    target = next(
        c for c in nb["cells"]
        if c.get("cell_type") == "code"
        and ('Both branches removed' in src(c) or 'hierarchical_null_embedding_sanity_check' in src(c))
    )
    s = src(target)
    old = 'for label,mode in [("Learned hierarchy","learned"),("Static branch removed","static_removed"),("Dynamic branch removed","dynamic_removed"),("Both branches removed","both_removed"),("Static branch shuffled","static_shuffled"),("Dynamic branch shuffled","dynamic_shuffled")]: rows.append({"intervention":label,**evaluate_hierarchical_intervention(model,test_loader,device,mode,seed)})'
    new = '''for label,mode in [("Learned hierarchy","learned"),("Static branch removed","static_removed"),("Dynamic branch removed","dynamic_removed"),("Static branch shuffled","static_shuffled"),("Dynamic branch shuffled","dynamic_shuffled")]: rows.append({"intervention":label,**evaluate_hierarchical_intervention(model,test_loader,device,mode,seed)})

# Both encoded branches set to zero is retained only as a null-input implementation check.
# Downstream biases and graph layers can still emit a prediction, so this is not a
# meaningful modality baseline and is excluded from the main comparison plot.
null_metrics = evaluate_hierarchical_intervention(model,test_loader,device,"both_removed",seed)
hierarchical_null_embedding_sanity_check = pd.DataFrame([{
    "condition":"Null-embedding sanity check",
    "remaining_computation":"fusion/joint-graph/output biases and transformations after zero branch embeddings",
    "scientific_use":"implementation sanity check only; not a modality baseline",
    **null_metrics,
}])
hierarchical_null_embedding_sanity_check.to_csv(
    os.path.join(output_folder,"hierarchical_null_embedding_sanity_check.csv"),index=False
)'''
    if old in s:
        s=s.replace(old,new)
    s=s.replace('"Occlusion-based static share"','"Occlusion-based static branch share"')
    s=s.replace('xlabel="Static effective share"','xlabel="Static occlusion-based branch share"')
    set_src(target,s)
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            t=src(c).replace('ylabel="Mean static share"','ylabel="Mean static occlusion share"')
            t=t.replace('ylabel="Static share"','ylabel="Static occlusion share"')
            t=t.replace('xlabel="Static effective share"','xlabel="Static occlusion-based branch share"')
            set_src(c,t)
    heading = next(c for c in nb["cells"] if c.get("cell_type") == "markdown" and "## H. Branch-removal" in src(c))
    if "Null-embedding" not in src(heading):
        set_src(heading, src(heading).rstrip()+'''\n\nThe main comparison includes branch removal and branch reassignment only. Setting both encoded branches to zero is retained separately as a **null-embedding sanity check**: downstream biases and transformations remain active, so its output is not a meaningful prediction baseline and must not be compared with independently trained models.\n''')
    for cell in nb["cells"]:
        text = src(cell).replace("M8 hierarchical fusion", "M33 hierarchical fusion")
        text = text.replace("Compare M8 directly with M7", "Compare M33 directly with M7")
        text = text.replace("M7 and M8 support", "M7 and M33 support")
        set_src(cell, text)


for filename in NOTEBOOKS:
    path = ROOT / filename
    nb = json.loads(path.read_text(encoding="utf-8"))
    model = filename.removesuffix(".ipynb")
    if not any("Shared publication plotting configuration" in src(c) for c in nb["cells"]):
        insert_at = next(i for i,c in enumerate(nb["cells"]) if c.get("cell_type") == "code" and "data_set =" in src(c)) + 1
        nb["cells"].insert(insert_at, code(STYLE_CELL))
    if filename.startswith("M7_"):
        complete_m7(nb)
    if filename.startswith("M33_"):
        complete_m33(nb)
        nb = replace_strings(nb, [
            ("M8 hierarchical fusion", "M33 hierarchical fusion"),
            ("Compare M8 directly with M7", "Compare M33 directly with M7"),
            ("M7 and M8 support", "M7 and M33 support"),
        ])
    add_interpretations(nb, model)
    add_traceability(nb, model)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    print(f"Updated {filename}")
