import os
import re
import unicodedata
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Marathi Movie Sentiment Analysis API",
    description="L3Cube-MahaSent Aspect-Based & Explainable Sentiment Engine",
    version="1.0.0"
)

# Enable CORS for React frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. Lexicons & Aspect Definitions (Zero Extra Dependencies)
# ---------------------------------------------------------
POSITIVE_WORDS = {
    "छान", "उत्कृष्ट", "सुंदर", "अप्रतिम", "मस्त", "सुरेख", "भारी", "खूप छान", 
    "आवडला", "विशेष", "सकारात्मक", "प्रेमळ", "उत्तम", "रोमांचक", "अविस्मरणीय", 
    "सर्वोत्कृष्ट", "विनोदी", "हसविणारा", "हसणारा", "दमदार", "प्रभावशाली"
}

NEGATIVE_WORDS = {
    "कंटाळवाणा", "बकवास", "वाईट", "भयानक", "व्यर्थ", "घाण", "फालतू", "कंटाळा", 
    "अतिशय वाईट", "वायफळ", "असमाधानकारक", "नाही आवडला", "निराशाजनक", "थकवणारा", 
    "चुकीचा", "अयशस्वी", "त्रुटी", "वेळ वाया", "पैसा वाया", "टुकार"
}

ASPECT_KEYWORDS = {
    "अभिनय (Acting)": ["अभिनय", "कलाकार", "अभिनेता", "अभिनेत्री", "काम", "पात्रे", "नायक", "नायिका"],
    "कथा (Story)": ["कथा", "गोष्ट", "पटकथा", "विषय", "संवाद", "प्लॉट", "मांडणी"],
    "संगीत (Music)": ["गाणी", "गाणे", "संगीत", "आवाज", "गायक", "स्वर", "पार्श्वसंगीत"],
    "दिग्दर्शन (Direction)": ["दिग्दर्शन", "दिग्दर्शक", "चित्रपट निर्मिती", "मेकिंग", "शैली"]
}

# ---------------------------------------------------------
# 2. Preprocessing & Feature Extraction Functions
# ---------------------------------------------------------
def clean_marathi_text(text: str) -> str:
    """NFC normalization and artifact cleanup matching training pipeline."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_aspects(text: str) -> dict:
    """Aspect-Based Sentiment Mining across the 4 cinematic pillars."""
    results = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        found = False
        for kw in keywords:
            if kw in text:
                found = True
                break
        
        if not found:
            results[aspect] = "Not Mentioned"
            continue
            
        pos_score = sum(1 for w in POSITIVE_WORDS if w in text)
        neg_score = sum(1 for w in NEGATIVE_WORDS if w in text)
        
        if pos_score > neg_score:
            results[aspect] = "Positive"
        elif neg_score > pos_score:
            results[aspect] = "Negative"
        else:
            results[aspect] = "Neutral"
            
    return results

def get_word_explainability(text: str) -> list:
    """Token-level polarity tags for visual XAI highlighting."""
    raw_tokens = re.findall(r"[\u0900-\u097F]+|[^\s\u0900-\u097F]+", text)
    annotated = []
    
    for token in raw_tokens:
        clean_tok = token.strip()
        if not clean_tok:
            continue
        if clean_tok in POSITIVE_WORDS:
            tag = "pos"
        elif clean_tok in NEGATIVE_WORDS:
            tag = "neg"
        else:
            tag = "neu"
        annotated.append({"word": clean_tok, "tag": tag})
        
    return annotated

# ---------------------------------------------------------
# 3. Dynamic Model Artifact Resolver
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

candidate_paths = [
    os.path.join(BASE_DIR, "models"),
    os.path.join(BASE_DIR, "..", "backend", "models"),
    os.path.join(os.getcwd(), "backend", "models"),
    os.path.join(os.getcwd(), "models")
]

vectorizer = None
model = None

for path in candidate_paths:
    vec_candidate = os.path.join(path, "tfidf_vectorizer.pkl")
    model_candidate = os.path.join(path, "sentiment_model.pkl")
    
    if os.path.exists(vec_candidate) and os.path.exists(model_candidate):
        try:
            vectorizer = joblib.load(vec_candidate)
            model = joblib.load(model_candidate)
            print(f"[SUCCESS] Loaded model artifacts from: {path}")
            break
        except Exception as e:
            print(f"[WARNING] Found files at {path} but failed to load: {e}")

if model is None or vectorizer is None:
    print("[INFO] Model files not found. Active mode: Robust Heuristic Engine.")

# Class label lookup
LABEL_MAP = {
    "Positive": ("Positive", "सकारात्मक"),
    "Negative": ("Negative", "नकारात्मक"),
    "Neutral":  ("Neutral",  "तटस्थ"),
    "1":        ("Positive", "सकारात्मक"),
    "-1":       ("Negative", "नकारात्मक"),
    "0":        ("Neutral",  "तटस्थ"),
    "2":        ("Positive", "सकारात्मक")
}

# ---------------------------------------------------------
# 4. API Schemas and Routes
# ---------------------------------------------------------
class ReviewRequest(BaseModel):
    review: str

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Marathi Sentiment Analysis API",
        "model_loaded": model is not None
    }

@app.post("/analyze")
def analyze(payload: ReviewRequest):
    raw_text = payload.review.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")

    cleaned = clean_marathi_text(raw_text)

    # Path A: Real ML Model Inference
    if vectorizer is not None and model is not None:
        try:
            X_vec = vectorizer.transform([cleaned])
            raw_pred = str(model.predict(X_vec)[0])
            
            sentiment, label_mr = LABEL_MAP.get(raw_pred, (raw_pred, raw_pred))
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_vec)[0]
                confidence = round(float(max(probs)) * 100, 1)
            else:
                confidence = 85.0
        except Exception as err:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(err)}")
            
    # Path B: Fallback Heuristic
    else:
        words = set(re.findall(r"[\u0900-\u097F]+", cleaned))
        pos_hits = len(words & POSITIVE_WORDS)
        neg_hits = len(words & NEGATIVE_WORDS)
        
        if pos_hits > neg_hits:
            sentiment, label_mr = "Positive", "सकारात्मक"
            confidence = round(78.0 + min(pos_hits * 3, 18), 1)
        elif neg_hits > pos_hits:
            sentiment, label_mr = "Negative", "नकारात्मक"
            confidence = round(78.0 + min(neg_hits * 3, 18), 1)
        else:
            sentiment, label_mr = "Neutral", "तटस्थ"
            confidence = 65.0

    return {
        "original_text": raw_text,
        "cleaned_text": cleaned,
        "sentiment": sentiment,
        "label_mr": label_mr,
        "confidence": confidence,
        "aspects": extract_aspects(cleaned),
        "tokens": get_word_explainability(cleaned)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
    