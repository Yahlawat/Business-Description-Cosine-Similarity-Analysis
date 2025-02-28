import torch
import pandas as pd
from sentence_transformers import util
from src.embedding import encode_sentences


def find_top_similar(input_texts, bd_embeddings, data, top_k=25):
    """Finds top-K most similar business descriptions."""
    input_embeddings = encode_sentences([t.lower() for t in input_texts])
    cosine_scores = util.cos_sim(input_embeddings, bd_embeddings)[0]

    top_results = torch.topk(cosine_scores, k=top_k)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        idx = idx.item()
        results.append(
            {
                "company_name": data.iloc[idx]["company_name"],
                "business_description": data.iloc[idx]["business_description"],
                "similar_sentence": data.iloc[idx]["individual_sentences"],
                "similarity_score": round(score.item(), 4),
            }
        )

    return pd.DataFrame(results)
