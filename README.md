# Business Description Clustering with SBERT

This project implements a natural language processing pipeline to cluster S&P 500 companies based on the semantic content of their business descriptions using SBERT embeddings and hierarchical clustering. The goal is to uncover functionally similar companies by analyzing their descriptions and grouping them according to operational or sector-based characteristics.

# <img src="./Images/image.png" alt="Business Description Clustering using NLP" width="400">

## Features

- Extracts and preprocesses business descriptions for S&P 500 companies
- Generates high-quality text embeddings using SBERT
- Applies dimensionality reduction techniques to streamline embeddings
- Performs hierarchical clustering to identify natural groupings and sectoral patterns
- Visualizes clustering results with interactive plots and dendrograms

## Prerequisites

- Python 3.12 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yahlawat/Business-Description-Clustering.git
cd Business-Description-Clustering
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the required spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Project Structure

```
Business-Description-Clustering/
├── notebooks/
│   └── Business_Description_Clustering.ipynb
├── requirements.txt
└── README.md
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook notebooks/Business_Description_Clustering.ipynb
```

2. Run all cells in sequence to:
   - Set up the environment and import required libraries
   - Collect S&P 500 company data using yfinance
   - Preprocess business descriptions
   - Generate SBERT embeddings
   - Apply dimensionality reduction
   - Perform hierarchical clustering
   - Visualize results

## Methodology

1. **Data Collection**: Retrieves S&P 500 company data and business descriptions using yfinance API
2. **Text Preprocessing**: Cleans and standardizes business descriptions
3. **Embedding Generation**: Uses SBERT model to create semantic embeddings
4. **Dimension Reduction**: Applies PCA and UMAP for efficient clustering
5. **Hierarchical Clustering**: Groups companies using Ward's linkage method
6. **Visualization**: Creates dendrograms and heatmaps to interpret results

## Dependencies

The project relies on several key libraries:
- pandas and numpy for data manipulation
- yfinance for financial data retrieval
- sentence-transformers for SBERT embeddings
- scikit-learn and scipy for clustering
- umap-learn for dimensionality reduction
- matplotlib and seaborn for visualization
- nltk and spacy for text processing

See `requirements.txt` for detailed version information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- S&P 500 companies data from Wikipedia
- Yahoo Finance API for business descriptions
- Sentence-BERT for semantic embeddings
- UMAP for dimensionality reduction 