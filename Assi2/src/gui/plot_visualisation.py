import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_matrix(data):
    plt.figure(figsize=(10, 6))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()
