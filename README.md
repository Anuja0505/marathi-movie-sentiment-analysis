# MahaSent-MR: Marathi Movie Sentiment Analysis

An end-to-end Machine Learning web application designed for Marathi Movie Review Sentiment Analysis, featuring granular Aspect-Based Sentiment Mining and Word-Level Explainability (XAI).

## 🚀 Key Features

- **Overall Sentiment Classification**: Classifies Marathi movie reviews into Positive (सकारात्मक), Negative (नकारात्मक), or Neutral (तटस्थ) with confidence scores.
- **Aspect-Based Sentiment Mining**: Clause-splitting engine to extract aspect-specific sentiments across 4 key cinematic pillars:
  - 🎭 **Acting (अभिनय)**
  - 📖 **Story (कथा)**
  - 🎵 **Music (संगीत)**
  - 🎬 **Direction (दिग्दर्शन)**
- **Word-Level Explainability (XAI)**: Highlights key positive and negative Marathi words contributing to the model's inference.

---

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, Scikit-learn, TF-IDF, Joblib, Pydantic
- **Frontend**: React, Vite, Tailwind CSS, Lucide Icons
- **Dataset**: L3Cube-MahaSent-MR / Custom Marathi Review Corpus

---

## 📂 Project Structure

```text
├── backend/            # FastAPI REST API & NLP Sentiment Engine
│   ├── models/         # Pre-trained ML artifacts (.pkl)
│   ├── app.py          # API endpoints & CORS configuration
│   └── engine.py       # Aspect mining & explainability logic
├── frontend-react/     # React + Vite interactive dashboard
├── dataset/            # Raw Marathi review datasets
├── preprocessing/      # Data cleaning and tokenization scripts
├── processed/          # Cleaned CSV datasets
└── training.py         # Model training script
