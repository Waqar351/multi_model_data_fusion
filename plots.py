
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np
from model_helper_funcs import get_neighbors

save_format = 'png'
dpi = 300
figsize = (10,8)

def plot_crime_vs_nocrime_ratio(data):
    # Calculate counts
    counts = data['crime_label'].value_counts()

    # Convert to percentages
    percentages = counts / counts.sum() * 100

    # Plot bar chart
    ax = percentages.plot.bar(color=['skyblue', 'salmon'])
    # ax = percentages.plot.bar(color=['green', 'red'])

    # Add percentage labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2, p.get_height()), 
                    ha='center', va='bottom')

    plt.ylabel('Percentage (%)')
    plt.title('Crime vs No Crime')
    plt.show()

def plot_train_val_curves(train_losses, val_accs, test_accs, filename = 'plot', output_path = None):
    # ==========================================================
    # Plot Training Curves
    # ==========================================================
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.plot(train_losses, label='Train Loss', color='royalblue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(val_accs, label='Validation Accuracy', color='darkorange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(test_accs, label='Test Accuracy', color='darkgreen')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy Curve')
    plt.legend()

    plt.tight_layout()

    # Save or show
    if output_path:
        file_path = os.path.join(output_path, f"{filename}.{save_format}")
        plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_embedding(emb_2d, crime_label_tensor, filename= 'embedding', output_path = None):
    df_emb = pd.DataFrame(emb_2d, columns=['x', 'y'])
    df_emb['label'] = crime_label_tensor.cpu().numpy()

    plt.figure(figsize=figsize)
    sns.scatterplot(data=df_emb, x='x', y='y', hue='label', palette='Set1')
    # Save or show
    if output_path:
        file_path = os.path.join(output_path, f"{filename}.{save_format}")
        plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_features_contribution(alpha_values, title = None , filename="plot_feature_contribution", output_path = None):
    """
    Plots the evolution of static and dynamic feature contributions (α values) over training epochs.

    Parameters
    ----------
    alpha_values : list
        Either a list of scalar α values (floats) or a list of dictionaries with keys 
        'static_dataset' and 'dynamic_dataset' per epoch.
    filename : str, optional
        Output filename (without extension).
    output_path : str, optional
        Directory to save the plot. If None, the plot is just displayed.
    save_format : str, optional
        File format to save the plot (default: 'png').
    dpi : int, optional
        Resolution of the saved figure (default: 300).
    """
    # epochs = np.arange(1, len(alpha_values) + 1)
    # alpha = np.array(alpha_values)
    # static_contrib = alpha
    # dynamic_contrib = 1 - alpha

    # --- Parse alpha values dynamically ---
    if isinstance(alpha_values[0], dict):
        static_contrib = [d.get("static_dataset", 0) for d in alpha_values]
        dynamic_contrib = [d.get("dynamic_dataset", 0) for d in alpha_values]
    else:
        static_contrib = np.array(alpha_values)
        dynamic_contrib = 1 - static_contrib
    
    epochs = np.arange(1, len(alpha_values) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, static_contrib, label='Static Contribution', linewidth=2)
    plt.plot(epochs, dynamic_contrib, label='Dynamic Contribution', linewidth=2, linestyle='--')

    # Highlight key thresholds
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    plt.axhline(0.5, color='lightgray', linestyle='--', linewidth=1)
    plt.axhline(1, color='gray', linestyle=':', linewidth=1)

    plt.title(f"{title}")
    plt.xlabel("Epoch")
    plt.ylabel("Contribution Weight")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # Save or show
    if output_path:
        file_path = os.path.join(output_path, f"{filename}.{save_format}")
        plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_gradient_contribution(alpha_values, title = None , filename="plot_feature_contribution", output_path = None):
    """
    Plots the evolution of static and dynamic feature contributions (α values) over training epochs.

    Parameters
    ----------
    alpha_values : list
        Either a list of scalar α values (floats) or a list of dictionaries with keys 
        'static_dataset' and 'dynamic_dataset' per epoch.
    filename : str, optional
        Output filename (without extension).
    output_path : str, optional
        Directory to save the plot. If None, the plot is just displayed.
    save_format : str, optional
        File format to save the plot (default: 'png').
    dpi : int, optional
        Resolution of the saved figure (default: 300).
    """
    # epochs = np.arange(1, len(alpha_values) + 1)
    # alpha = np.array(alpha_values)
    # static_contrib = alpha
    # dynamic_contrib = 1 - alpha

    # --- Parse alpha values dynamically ---
    if isinstance(alpha_values[0], dict):
        static_contrib = [d.get("grad_ratio_static", 0) for d in alpha_values]
        dynamic_contrib = [d.get("grad_ratio_dynamic", 0) for d in alpha_values]
    else:
        static_contrib = np.array(alpha_values)
        dynamic_contrib = 1 - static_contrib
    
    epochs = np.arange(1, len(alpha_values) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, static_contrib, label='grad_st Contribution', linewidth=2)
    plt.plot(epochs, dynamic_contrib, label='grad_dy Contribution', linewidth=2, linestyle='--')

    # Highlight key thresholds
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    plt.axhline(0.5, color='lightgray', linestyle='--', linewidth=1)
    plt.axhline(1, color='gray', linestyle=':', linewidth=1)

    plt.title(f"{title}")
    plt.xlabel("Epoch")
    plt.ylabel("Contribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # Save or show
    if output_path:
        file_path = os.path.join(output_path, f"{filename}.{save_format}")
        plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()
    
def plot_feature_contr_validation_scr(alpha_values, val_accs, title = None, filename = "feature_contrib_val_scr", output_path = None):

    # epochs = np.arange(1, len(alpha_values) + 1)
    # alpha = np.array(alpha_values)
    # static_contrib = alpha
    # dynamic_contrib = 1 - alpha
    val_acc = np.array(val_accs)
    # --- Parse alpha values dynamically ---
    if isinstance(alpha_values[0], dict):
        static_contrib = [d.get("static_dataset", 0) for d in alpha_values]
        dynamic_contrib = [d.get("dynamic_dataset", 0) for d in alpha_values]
    else:
        static_contrib = np.array(alpha_values)
        dynamic_contrib = 1 - static_contrib
    
    epochs = np.arange(1, len(alpha_values) + 1)

    # --- Create figure ---
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Plot α (static) and (1-α) (dynamic) on the left y-axis
    ax1.plot(epochs, static_contrib, label='Static Contribution', color='tab:blue', linewidth=2)
    ax1.plot(epochs, dynamic_contrib, label='Dynamic Contribution', color='tab:orange', linewidth=2, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Contribution', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(alpha=0.3)

    # Highlight baseline levels
    ax1.axhline(0.5, color='lightgray', linestyle='--', linewidth=1)
    ax1.axhline(0, color='gray', linestyle=':', linewidth=1)

    # --- Add validation accuracy on secondary y-axis ---
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_acc, label='Validation Accuracy', color='tab:green', linewidth=2.5)
    ax2.set_ylabel('Validation Accuracy', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # --- Combine legends from both axes ---
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')

    plt.title(f"{title}")
    plt.tight_layout()
    
    # Save or show
    if output_path:
        file_path = os.path.join(output_path, f"{filename}.{save_format}")
        plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()


#------------------- START: Time Series Analysis_________________________________________________

import matplotlib.pyplot as plt
import numpy as np


# def plot_fluctuation_analysis(
#     pred_matrix,
#     node,
#     adj,
#     max_hop=2,
#     mode="mean",  # "mean" or "all"
#     output_path = None,
#     filename = None
# ):
#     """
#     pred_matrix : shape (T, N)
#     node        : central node id
#     adj         : adjacency dictionary
#     max_hop     : how many hops to include
#     mode        : "mean" or "all"
#     """

#     T = pred_matrix.shape[0]
#     time = np.arange(T)

#     plt.figure(figsize=(12, 6))

#     # ---- Central node ----
#     central = pred_matrix[:, node]
#     plt.plot(
#         time,
#         central,
#         linewidth=3,
#         color="black",
#         label=f"Node {node} (Central)"
#     )

#     # Color map for hops
#     hop_colors = plt.cm.viridis(np.linspace(0.3, 0.9, max_hop))

#     # ---- For each hop ----
#     for hop in range(1, max_hop + 1):

#         neighbors = get_neighbors(node, adj, order=hop)

#         if len(neighbors) == 0:
#             continue

#         color = hop_colors[hop - 1]

#         if mode == "mean":
#             hop_series = pred_matrix[:, neighbors].mean(axis=1)

#             plt.plot(
#                 time,
#                 hop_series,
#                 linewidth=2,
#                 linestyle="--",
#                 color=color,
#                 label=f"Mean {hop}-hop Neighbors"
#             )

#         elif mode == "all":
#             # Plot all neighbors but same color
#             for n in neighbors:
#                 plt.plot(
#                     time,
#                     pred_matrix[:, n],
#                     linewidth=1.5,
#                     alpha=0.5,
#                     color=color
#                 )

#             # Add only one legend entry per hop
#             plt.plot(
#                 [],
#                 [],
#                 color=color,
#                 linewidth=2,
#                 label=f"{hop}-hop Neighbors"
#             )

#     plt.xlabel("Time Step")
#     plt.ylabel("Predicted Crime Intensity")
#     plt.title("Spatio-Temporal Crime Evolution")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     # plt.show()

#     # Save or show
#     if output_path:
#         file_path = os.path.join(output_path, f"{filename}.{save_format}")
#         plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
#         plt.show()
#         plt.close()
#     else:
#         plt.show()
#         plt.close()

def plot_fluctuation_analysis(
    pred_matrix,
    node,
    adj,
    max_hop=2,
    mode="mean",
    output_path=None,
    filename=None,
    show_correlation=True
):
    """
    pred_matrix : shape (T, N)
    node        : central node id
    adj         : adjacency dictionary
    max_hop     : number of hops
    mode        : "mean" or "all"
    show_correlation : display correlation values in plot
    """

    T = pred_matrix.shape[0]
    time = np.arange(T)

    plt.figure(figsize=(8, 5))

    # ---- Central node ----
    central = pred_matrix[:, node]
    plt.plot(
        time,
        central,
        linewidth=3,
        color="black",
        label=f"Node {node} (Central)"
    )

    hop_colors = plt.cm.viridis(np.linspace(0.3, 0.9, max_hop))

    correlations = []

    for hop in range(1, max_hop + 1):

        neighbors = get_neighbors(node, adj, order=hop)

        if len(neighbors) == 0:
            continue

        color = hop_colors[hop - 1]

        # Mean series for hop
        hop_series = pred_matrix[:, neighbors].mean(axis=1)

        # ---- Correlation ----
        corr = np.corrcoef(central, hop_series)[0, 1]
        correlations.append((hop, corr))

        if mode == "mean":

            plt.plot(
                time,
                hop_series,
                linewidth=2,
                linestyle="--",
                color=color,
                label=f"Mean {hop}-hop (ρ={corr:.2f})"
            )

        elif mode == "all":

            for n in neighbors:
                plt.plot(
                    time,
                    pred_matrix[:, n],
                    linewidth=1.5,
                    alpha=0.4,
                    color=color
                )

            # One legend entry
            plt.plot(
                [],
                [],
                color=color,
                linewidth=2,
                label=f"{hop}-hop (ρ={corr:.2f})"
            )

    plt.xlabel("Time Step")
    plt.ylabel("Predicted Crime Intensity")
    plt.title("Spatio-Temporal Crime Evolution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Optional: correlation text box
    if show_correlation and correlations:
        corr_text = "\n".join(
            [f"{hop}-hop ρ = {corr:.3f}" for hop, corr in correlations]
        )
        plt.gcf().text(
            0.85, 0.75,
            corr_text,
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )

    if output_path:
        file_path = os.path.join(output_path, f"{filename}.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    else:
        plt.show()
        plt.close()


