import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)


def load_data(file_path):
    """Loads and preprocesses business description data from a CSV file."""
    df = pd.read_csv(file_path)
    df.rename(
        columns={
            "Company Name": "company_name",
            "Business Description": "business_description",
        },
        inplace=True,
    )
    df["cleaned_description"] = df["business_description"].str.lower()
    df["sentences"] = df["cleaned_description"].apply(sent_tokenize)

    df_expanded = (
        df.explode("sentences")
        .rename(columns={"sentences": "individual_sentences"})
        .reset_index(drop=True)
    )
    return df_expanded
