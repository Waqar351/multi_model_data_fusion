
import matplotlib.pyplot as plt

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