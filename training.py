import os
import glob
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Define paths relative to the project root directory
PROCESSED_DIR = os.path.join(".", "processed")
OUTPUT_MODEL_DIR = os.path.join(".", "backend", "models")

def get_processed_filepath(pattern):
    files = glob.glob(os.path.join(PROCESSED_DIR, pattern))
    if not files:
        raise FileNotFoundError(f"No file matching '{pattern}' found in '{PROCESSED_DIR}' folder.")
    return files[0]

# 2. Locate preprocessed files
train_path = get_processed_filepath('*Train*')
val_path = get_processed_filepath('*Val*')
test_path = get_processed_filepath('*Test*')

print("Loading preprocessed datasets:")
print(f" - Train: {train_path}")
print(f" - Val:   {val_path}")
print(f" - Test:  {test_path}\n")

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

# 3. Handle column names automatically
text_col = 'clean_text' if 'clean_text' in train_df.columns else 'marathi_sentence'
label_col = 'label'

# Ensure string types and handle null values
train_df[text_col] = train_df[text_col].fillna("").astype(str)
val_df[text_col] = val_df[text_col].fillna("").astype(str)
test_df[text_col] = test_df[text_col].fillna("").astype(str)

# Map labels (-1, 0, 1) to readable string classes
label_map = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
y_train = train_df[label_col].map(label_map)
y_val = val_df[label_col].map(label_map)
y_test = test_df[label_col].map(label_map)

# 4. Extract TF-IDF Features
print("1. Extracting TF-IDF Features...")
vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train_df[text_col])
X_val = vectorizer.transform(val_df[text_col])
X_test = vectorizer.transform(test_df[text_col])

# 5. Train Logistic Regression
print("2. Training Logistic Regression Classifier...")
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=500,
    C=1.0
)
model.fit(X_train, y_train)

# 6. Evaluate Performance
print("\n================ VALIDATION PERFORMANCE ================")
y_val_pred = model.predict(X_val)
print(classification_report(y_val, y_val_pred, digits=4))

print("\n================ FINAL TEST PERFORMANCE ================")
y_test_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {acc * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_test_pred, digits=4))

print("Confusion Matrix:")
labels = ['Positive', 'Neutral', 'Negative']
cm = confusion_matrix(y_test, y_test_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
print(cm_df)

# 7. Save Models directly into backend/models/
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

vec_path = os.path.join(OUTPUT_MODEL_DIR, 'tfidf_vectorizer.pkl')
model_path = os.path.join(OUTPUT_MODEL_DIR, 'sentiment_model.pkl')

joblib.dump(vectorizer, vec_path)
joblib.dump(model, model_path)

print(f"\nModel artifacts successfully saved to:")
print(f" - {vec_path}")
print(f" - {model_path}")
