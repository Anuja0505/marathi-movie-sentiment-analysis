import pandas as pd
import re


TRAIN_FILE = "processed/MahaSent_MR_Train_Preprocessed.csv"
VAL_FILE = "processed/MahaSent_MR_Val_Preprocessed.csv"
TEST_FILE = "processed/MahaSent_MR_Test_Preprocessed.csv"


def analyze_dataset(file_path, dataset_name):
    df = pd.read_csv(file_path)

    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)

    # Basic information
    print("\nShape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    # Missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Duplicate reviews
    print("\nDuplicate reviews:")
    print(df["marathi_sentence"].duplicated().sum())

    # Label distribution
    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

    # Review length
    df["word_count"] = df["cleaned_sentence"].str.split().str.len()

    print("\nReview length statistics:")
    print(df["word_count"].describe())

    # Empty cleaned reviews
    empty_reviews = (
        df["cleaned_sentence"]
        .str.strip()
        .eq("")
        .sum()
    )

    print("\nEmpty cleaned reviews:", empty_reviews)

    # URLs remaining
    urls = df["cleaned_sentence"].str.contains(
        r"https?://|www\.",
        regex=True,
        na=False
    ).sum()

    print("Reviews containing URLs:", urls)

    # HTML remaining
    html = df["cleaned_sentence"].str.contains(
        r"<[^>]+>",
        regex=True,
        na=False
    ).sum()

    print("Reviews containing HTML:", html)

    # Numbers
    numbers = df["cleaned_sentence"].str.contains(
        r"\d",
        regex=True,
        na=False
    ).sum()

    print("Reviews containing numbers:", numbers)

    # English alphabet characters
    english = df["cleaned_sentence"].str.contains(
        r"[A-Za-z]",
        regex=True,
        na=False
    ).sum()

    print("Reviews containing English characters:", english)

    # Devanagari characters
    devanagari = df["cleaned_sentence"].str.contains(
        r"[\u0900-\u097F]",
        regex=True,
        na=False
    ).sum()

    print("Reviews containing Devanagari:", devanagari)

    # Special characters
    special = df["cleaned_sentence"].str.contains(
        r"[^\w\s\u0900-\u097F]",
        regex=True,
        na=False
    ).sum()

    print("Reviews containing special characters:", special)

    # Show examples
    print("\nSample reviews:")
    print(
        df[
            ["marathi_sentence", "cleaned_sentence", "label"]
        ].head(10).to_string(index=False)
    )

    # Longest reviews
    print("\nLongest reviews:")
    longest = df.nlargest(5, "word_count")

    print(
        longest[
            ["cleaned_sentence", "word_count", "label"]
        ].to_string(index=False)
    )


# Analyze all three datasets

analyze_dataset(
    TRAIN_FILE,
    "TRAIN DATASET"
)

analyze_dataset(
    VAL_FILE,
    "VALIDATION DATASET"
)

analyze_dataset(
    TEST_FILE,
    "TEST DATASET"
)