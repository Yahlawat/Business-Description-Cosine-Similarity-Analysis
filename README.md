# Business Description Clustering

This project clusters companies based on their business descriptions using **Sentence Transformers** and **Cosine Similarity**. It helps in identifying similar businesses based on their descriptions.

---

## 📌 Project Overview
### Steps in the pipeline:
1. **Load and Preprocess Data**  
   - Read business descriptions from a CSV file.  
   - Convert descriptions to lowercase and tokenize them into sentences.

2. **Encode Descriptions Using Sentence Transformers**  
   - Convert business descriptions into high-dimensional embeddings using `all-mpnet-base-v2`.

3. **Compute Similarity Scores**  
   - Compare business descriptions using cosine similarity.
   - Retrieve the **top 25 most similar companies**.

4. **Filter Non-Similar Companies**  
   - Exclude companies that match a predefined **non-relevant category** (e.g., loan providers).

5. **Visualize Results**  
   - Generate a **heatmap** of similarity scores between companies.

6. **Save the Results**  
   - Store **filtered** and **unfiltered** similarity results in CSV format.

---

## 📂 Project Structure
```
business-description-clustering/
│── data/
│   ├── raw/                      # Original dataset
│   │   ├── management_support.csv
│   ├── results/                   # Final outputs
│── notebooks/                      # Jupyter notebooks for EDA
│── src/
│   ├── data_preprocessing.py       # Data cleaning & processing
│   ├── embedding.py                # Sentence embedding with transformers
│   ├── similarity.py               # Similarity search functions
│   ├── filtering.py                # Filtering non-relevant companies
│   ├── visualization.py            # Heatmap visualization
│── main.py                          # Main execution script
│── requirements.txt                 # Dependencies list
│── README.md                        # Project documentation
```

---

## 📦 Dependencies
This project requires **Python 3.12**. Install the required libraries using:
```bash
pip install -r requirements.txt
```

### Key Libraries:
- `pandas` → Data processing
- `numpy` → Numerical operations
- `sentence-transformers` → Text embeddings
- `nltk` → Text tokenization
- `torch` → Deep learning framework
- `seaborn` → Data visualization

---

## 🚀 Running the Project
### 1️⃣ Running via Jupyter Notebook
Execute the **Jupyter Notebook** step by step:
```bash
jupyter notebook
```
Then, open and run **Business_Description_Clustering.ipynb**.

### 2️⃣ Running via Python Script
For **command-line execution**, run:
```bash
python main.py
```

---

## 📊 Example Output
### 🔹 Sample Similarity Results
| Company Name      | Business Description                        | Similar Sentence             | Similarity Score |
|------------------|--------------------------------------------|-----------------------------|------------------|
| ABC Consulting  | Provides financial advisory services.       | Provides consultancy services. | 0.92            |
| XYZ Ltd.        | IT solutions and consulting for enterprises. | IT consultancy.              | 0.89            |

### 🔹 Heatmap of Similarity Scores
A **heatmap** is generated to show the **relationship between similar companies**.

---

## 📜 License
This project is licensed under the **MIT License**.

---
