import torch
from sentence_transformers import util
from src.embedding import encode_sentences


def is_similar_to_non_similar(
    description_embedding, non_similar_embeddings, threshold=0.6
):
    """Determines if a sentence is similar to a predefined set of non-similar sentences."""
    cosine_scores = util.cos_sim(description_embedding, non_similar_embeddings)[0]
    return cosine_scores.max().item() >= threshold


def filter_non_similar(df_expanded, non_similar_sentences):
    """Filters out companies whose descriptions are too similar to non-relevant categories."""
    non_similar_embeddings = encode_sentences(non_similar_sentences)
    companies_to_exclude = set()

    grouped = df_expanded.groupby("company_name")
    for company_name, group in grouped:
        embeddings = encode_sentences(group["individual_sentences"].tolist())
        for embedding in embeddings:
            if is_similar_to_non_similar(embedding, non_similar_embeddings):
                companies_to_exclude.add(company_name)
                break

    return df_expanded[
        ~df_expanded["company_name"].isin(companies_to_exclude)
    ].reset_index(drop=True)
