from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import joblib

from engine import (
    preprocess_text,
    analyze_aspects,
    get_word_explainability,
    POSITIVE_WORDS,
    NEGATIVE_WORDS
)

app = FastAPI(title="Marathi Movie Review Sentiment Analysis", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LABEL_MAP = {
    "1": {"sentiment": "Positive", "label_mr": "सकारात्मक"},
    "0": {"sentiment": "Neutral", "label_mr": "तटस्थ"},
    "-1": {"sentiment": "Negative", "label_mr": "नकारात्मक"},
    "2": {"sentiment": "Positive", "label_mr": "सकारात्मक"}
}

MODEL_PATH = os.path.join("..", "models", "sentiment_pipeline.pkl")
pipeline = None

if os.path.exists(MODEL_PATH):
    try:
        pipeline = joblib.load(MODEL_PATH)
        print("Trained model pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading model pipeline: {e}. Running in heuristic fallback mode.")

class ReviewRequest(BaseModel):
    review: str

@app.post("/analyze")
def analyze_review(payload: ReviewRequest):
    raw_text = payload.review.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")

    cleaned = preprocess_text(raw_text)

    # 1. Model Inference (or Heuristic Fallback)
    if pipeline is not None:
        try:
            pred = str(pipeline.predict([cleaned])[0])
            meta = LABEL_MAP.get(pred, {"sentiment": "Neutral", "label_mr": "तटस्थ"})
            sentiment = meta["sentiment"]
            label_mr = meta["label_mr"]

            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba([cleaned])[0]
                confidence = round(float(max(probs)) * 100, 1)
            else:
                confidence = 90.0
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
    else:
        # Development fallback based on positive and negative cues
        words = set(re.findall(r"[\u0900-\u097F]+", cleaned))
        pos_hits = len(words & POSITIVE_WORDS)
        neg_hits = len(words & NEGATIVE_WORDS)

        if pos_hits > neg_hits:
            sentiment, label_mr = "Positive", "सकारात्मक"
            confidence = round(82.0 + min(pos_hits * 3, 15), 1)
        elif neg_hits > pos_hits:
            sentiment, label_mr = "Negative", "नकारात्मक"
            confidence = round(82.0 + min(neg_hits * 3, 15), 1)
        else:
            sentiment, label_mr = "Neutral", "तटस्थ"
            confidence = 68.5

    # 2. Extract Features
    aspects = analyze_aspects(cleaned)
    tokens = get_word_explainability(cleaned)

    return {
        "original_text": raw_text,
        "cleaned_text": cleaned,
        "sentiment": sentiment,
        "label_mr": label_mr,
        "confidence": confidence,
        "aspects": aspects,
        "tokens": tokens
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)