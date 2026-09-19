import pandas as pd
import re
import unicodedata


INPUT_FILE = "dataset/MahaSent_MR_Val.csv"
OUTPUT_FILE = "processed/MahaSent_MR_Val_Preprocessed.csv"


def preprocess_text(text):
    # Convert to string
    text = str(text)

    # 1. Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove HTML tags
    text = re.sub(r"<[^>]*>", " ", text)

    # 4. Remove common HTML entities
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)

    # 5. Remove zero-width and invisible characters
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", text)

    # 6. Replace line breaks/tabs with spaces
    text = re.sub(r"[\r\n\t]+", " ", text)

    # 7. Normalize multiple spaces
    text = re.sub(r"\s+", " ", text)

    # 8. Remove leading/trailing whitespace
    text = text.strip()

    return text

# Load validation dataset
df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)
print("\nOriginal columns:")
print(df.columns)

# Remove unnecessary index column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Remove missing reviews or labels
df = df.dropna(subset=["marathi_sentence", "label"])

# Remove duplicate reviews
df = df.drop_duplicates(subset=["marathi_sentence"])

# Apply preprocessing
df["cleaned_sentence"] = df["marathi_sentence"].apply(preprocess_text)

# Remove empty reviews
df = df[df["cleaned_sentence"].str.strip() != ""]

# Save processed dataset
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\nProcessed shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nSample:")
print(df[["marathi_sentence", "cleaned_sentence", "label"]].head())

print("\nPreprocessed validation dataset saved successfully.")