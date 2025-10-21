# notebook.py — Amazon Fine Food Reviews Rating Predictor (Robust Version)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# --- NLTK Downloads ---
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Text cleaning with negation handling ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\bnot (\w+)\b', r'not_\1', text)  # handle negations
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    cleaned = " ".join(words)
    cleaned = cleaned.replace(" ", "_")  # convert spaces to underscores for keyword matching
    return cleaned

# --- Load dataset ---
data_path = "data/amazon_reviews.csv"
df = pd.read_csv(data_path)
print("✅ Dataset loaded successfully! Shape:", df.shape)

df = df[["Text", "Score"]].dropna()
df = df.rename(columns={"Text": "review_text", "Score": "rating"})
df['review_text'] = df['review_text'].apply(clean_text)

# --- Optional subsample for speed ---
df = df.sample(10000, random_state=42)

# --- Class balancing ---
min_count = df['rating'].value_counts().min()
df_balanced = pd.concat([
    df[df['rating'] == rating].sample(min_count, random_state=42)
    for rating in df['rating'].unique()
])

print("✅ Balanced dataset shape:", df_balanced.shape)
X = df_balanced["review_text"].values
y = df_balanced["rating"].astype(int).values

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- TF-IDF + Logistic Regression ---
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=300, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test_tfidf)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# --- Keyword overrides ---
negative_keywords = [
    "horrible", "rude", "dirty", "never", "cold", "worst", "bad", "terrible",
    "awful", "disgusting", "stale", "salty", "oily", "greasy", "burnt",
    "uncooked", "undercooked", "overcooked", "tasteless", "not_fresh",
    "not_tasty", "low_quality", "poor", "slow_service", "bland", "unhygienic",
    "smelly", "hard", "dry", "soggy", "served_late", "delayed_service",
    "long_wait", "staff_rude", "inattentive_staff", "overpriced"
]

moderate_keywords = [
    "okay", "average", "fine", "decent", "not_bad", "could_be_better", "nothing_special"
]

# convert spaces to underscores for matching
negative_keywords = [kw.replace(" ", "_") for kw in negative_keywords]
moderate_keywords = [kw.replace(" ", "_") for kw in moderate_keywords]

def keyword_override(text):
    text_cleaned = clean_text(text)
    for word in negative_keywords:
        if word in text_cleaned:
            return 2  # mild negative → 2 stars
    for word in moderate_keywords:
        if word in text_cleaned:
            return 3
    return None

def adjust_prediction(review_text):
    override = keyword_override(review_text)
    if override is not None:
        return override
    review_vector = vectorizer.transform([clean_text(review_text)])
    pred = model.predict(review_vector)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(review_vector)[0]
        if pred == 5 and proba[4] < 0.6:
            if proba[3] > 0.25: pred = 4
            elif proba[2] > 0.25: pred = 3
        elif pred == 4 and proba[3] < 0.4:
            pred = 3
        elif pred == 3 and (proba[2] > 0.35 or proba[1] > 0.25):
            pred = 2
    return pred

# --- Save model & vectorizer ---
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
joblib.dump(vectorizer, "models/vectorizer.joblib")
print("\n🎉 Model and vectorizer saved successfully in /models/")
