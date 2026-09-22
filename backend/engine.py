import re
import unicodedata

# Preprocessing matching teammate's script character-for-character
def preprocess_text(text: str) -> str:
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
    return text.strip()

# Domain polarity lexicons for explanation & aspect scoring
POSITIVE_WORDS = {
    "छान", "उत्कृष्ट", "सुंदर", "भारी", "मस्त", "अप्रतिम", "उत्तम",
    "आवडला", "आवडली", "प्रभावशाली", "यशस्वी", "मजेदार", "रोमांचक", "हृदयस्पर्शी", "जबरदस्त"
}

NEGATIVE_WORDS = {
    "वाईट", "बकवास", "कंटाळवाणा", "कंटाळवाणी", "निराशाजनक", "थकवणारा", "चुकीचा",
    "भिकार", "घाण", "फालतू", "रटाळ", "अतिशयोक्ती", "व्यर्थ", "वाया", "असमाधानकारक"
}

ASPECT_KEYWORDS = {
    "अभिनय (Acting)": ["अभिनय", "कलाकार", "नायक", "नायिका", "पात्र", "काम", "अभिनेता"],
    "कथा (Story)": ["कथा", "गोष्ट", "पटकथा", "संवाद", "लेखन", "प्लॉट"],
    "संगीत (Music)": ["संगीत", "गाणी", "गाणं", "गीत", "गायक", "गायन", "सूर", "धुन"],
    "दिग्दर्शन (Direction)": ["दिग्दर्शन", "दिग्दर्शक", "मेकिंग", "दिग्दर्शकाने"]
}

def analyze_aspects(cleaned_text: str) -> dict:
    """Evaluates sentiment per film aspect by analyzing related clauses."""
    clauses = re.split(r"[,;।|]|\s+(?:पण|परंतु|आणि|तसेच|मात्र)\s+", cleaned_text)
    aspect_scores = {}

    for aspect, keywords in ASPECT_KEYWORDS.items():
        pos_hits = 0
        neg_hits = 0
        mentioned = False

        for clause in clauses:
            clause = clause.strip()
            if any(k in clause for k in keywords):
                mentioned = True
                words = set(re.findall(r"[\u0900-\u097F]+", clause))
                pos_hits += len(words & POSITIVE_WORDS)
                neg_hits += len(words & NEGATIVE_WORDS)

        if mentioned:
            if pos_hits > neg_hits:
                aspect_scores[aspect] = "Positive"
            elif neg_hits > pos_hits:
                aspect_scores[aspect] = "Negative"
            else:
                aspect_scores[aspect] = "Neutral"
        else:
            aspect_scores[aspect] = "Not Mentioned"

    return aspect_scores

def get_word_explainability(cleaned_text: str):
    """Tags individual words for explainability highlighting in the UI."""
    tokens = []
    for word in cleaned_text.split():
        clean_word = re.sub(r"[^\u0900-\u097F]", "", word)
        if clean_word in POSITIVE_WORDS:
            tokens.append({"word": word, "tag": "pos"})
        elif clean_word in NEGATIVE_WORDS:
            tokens.append({"word": word, "tag": "neg"})
        else:
            tokens.append({"word": word, "tag": "neu"})
    return tokens