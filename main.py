import os
import pandas as pd
from src.data_preprocessing import load_data
from src.embedding import encode_sentences
from src.similarity import find_top_similar
from src.filtering_non_similar_companies import filter_non_similar
from src.visualization import plot_similarity_heatmap

# Set up paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data", "raw", "management_support.csv")
OUTPUT_PATH = os.path.join(BASE_PATH, "data", "results")

# Load and preprocess data
print("\nLoading and Preprocessing Data...")
df_expanded = load_data(DATA_PATH)

# Encode business descriptions
print("\nEncoding Business Descriptions...")
bd_sentences = df_expanded["individual_sentences"].tolist()
bd_embeddings = encode_sentences(bd_sentences)

# Find similar companies
print("\nFinding Similar Companies...")
input_sentences = ["provides consultancy services."]
results_df = find_top_similar(input_sentences, bd_embeddings, df_expanded)
results_df.to_csv(os.path.join(OUTPUT_PATH, "similar_companies.csv"), index=False)

# Filter out non-relevant companies
print("\nFiltering Non-Relevant Companies...")
non_similar_sentences = ["company provides consumer loans"]
df_filtered = filter_non_similar(df_expanded, non_similar_sentences)

# Find similar companies after filtering
filtered_results_df = find_top_similar(
    input_sentences,
    encode_sentences(df_filtered["individual_sentences"].tolist()),
    df_filtered,
)
filtered_results_df.to_csv(
    os.path.join(OUTPUT_PATH, "filtered_companies.csv"), index=False
)

# Plot similarity heatmap
plot_similarity_heatmap(filtered_results_df)

print("\nProcessing Complete. ✅")
