# Business Description Clustering

A machine learning pipeline that clusters companies based on their business descriptions using FinBERT embeddings and cosine similarity.

## Overview

This project uses natural language processing to identify similar companies based on their business descriptions. It leverages FinBERT, a financial domain-specific BERT model, to generate high-quality embeddings that capture the semantic meaning of business descriptions.

### Key Features
- Text preprocessing with case preservation for financial terms
- Sentence embeddings using FinBERT
- Business function detection and filtering
- Similarity scoring with configurable threshold
- Results export in CSV format

## Project Structure
```
business-description-clustering/
├── data/
│   ├── raw/              # Original dataset
│   │   └── management_support.csv
│   └── results/          # Generated outputs
│       └── filtered_companies.csv
├── notebooks/            # Jupyter notebooks
│   └── Business_Description_FinBERT.ipynb
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Git ignore rules
```

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\\Scripts\\activate   # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Place your company data CSV in `data/raw/` with columns:
   - Company Name
   - Business Description

2. Run the Jupyter notebook:
```bash
jupyter notebook notebooks/Business_Description_FinBERT.ipynb
```

3. Results will be saved to `data/results/filtered_companies.csv`

## Configuration

Key parameters in the notebook:
- `SIMILARITY_THRESHOLD`: Minimum similarity score (default: 0.75)
- `FUNCTION_KEYWORDS`: Keywords for business function detection
- `MODEL_NAME`: FinBERT model variant (default: "yiyanghkust/finbert-tone")

## Dependencies

- Python 3.12+
- PyTorch
- Transformers (Hugging Face)
- spaCy
- pandas
- NLTK

See `requirements.txt` for complete list with versions.

## License

This project is licensed under the MIT License.
