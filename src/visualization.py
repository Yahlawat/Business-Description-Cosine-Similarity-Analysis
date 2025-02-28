import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_similarity_heatmap(results_df):
    """Plots a heatmap of similarity scores between companies."""
    pivot_table = results_df.pivot(
        index="company_name", columns="similar_sentence", values="similarity_score"
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, cmap="coolwarm", annot=False)
    plt.title("Company Similarity Heatmap")
    plt.xlabel("Similar Sentences")
    plt.ylabel("Company Name")
    plt.show()
