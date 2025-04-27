# Business Description Semantic Search

A machine learning pipeline to retrieve functionally similar companies based on business descriptions using SBERT embeddings and cosine similarity analysis.

<img src=./Images/image.png alt="" width="300">

## Overview

This project applies natural language processing to identify companies with similar operational profiles by:

- Generating dense semantic embeddings of business descriptions using **SBERT (multi-qa-mpnet-base-cos-v1)**.
- Performing **prompt-based retrieval** with cosine similarity scoring.
- Grouping and ranking companies based on **average functional similarity**.
- Visualizing the Top-N most similar companies.

## Key Features

- Text cleaning and functional sentence filtering.
- High-quality sentence embeddings optimized for semantic search.
- Top-N retrieval of companies based on functional prompts.
- Batch processing for scalable embedding generation.
- Results export and easy visualization.

## Dependencies

- Python 3.10+
- PyTorch
- sentence-transformers
- Transformers (Hugging Face)
- spaCy
- pandas
- NLTK

See `requirements.txt` for full package versions.

## License

This project is licensed under the MIT License.
