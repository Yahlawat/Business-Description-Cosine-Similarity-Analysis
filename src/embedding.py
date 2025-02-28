import torch
from sentence_transformers import SentenceTransformer

# Load model once globally
model = SentenceTransformer("all-mpnet-base-v2")


def encode_sentences(sentences):
    """Encodes a list of sentences using the Sentence Transformer model."""
    return model.encode(sentences, convert_to_tensor=True)
