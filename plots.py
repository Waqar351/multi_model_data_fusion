
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np

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

def plot_features_contribution(alpha_values, filename="plot_feature_contribution", output_path = None):
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
    plt.plot(epochs, static_contrib, label='Static Contribution (α)', linewidth=2)
    plt.plot(epochs, dynamic_contrib, label='Dynamic Contribution (1−α)', linewidth=2, linestyle='--')

    # Highlight key thresholds
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    plt.axhline(0.5, color='lightgray', linestyle='--', linewidth=1)
    plt.axhline(1, color='gray', linestyle=':', linewidth=1)

    plt.title("Evolution of Fusion Weights (α) Over Training")
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
    
def plot_feature_contr_validation_scr(alpha_values, val_accs, filename = "feature_contrib_val_scr", output_path = None):

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
    ax1.plot(epochs, static_contrib, label='Static Contribution (α)', color='tab:blue', linewidth=2)
    ax1.plot(epochs, dynamic_contrib, label='Dynamic Contribution (1−α)', color='tab:orange', linewidth=2, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Fusion Coefficient', color='tab:blue')
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

    plt.title("Fusion Behavior vs Model Performance")
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
