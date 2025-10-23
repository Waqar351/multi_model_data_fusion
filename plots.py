
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

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