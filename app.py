# app.py — Streamlit Web App (Final Version)
import streamlit as st
import joblib
import re
import nltk

# --- Step 1: Download NLTK data before importing from them ---
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation & numbers
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# --- Step 2: Load trained model & vectorizer ---
model = joblib.load("models/model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

# --- Step 3: Define negative keyword override ---
negative_keywords = [
    "horrible", "rude", "dirty", "never", "cold", "worst", "bad", "terrible", "awful"
]

def keyword_override(text):
    """Force 1-star if obvious negative words are present"""
    for word in negative_keywords:
        if word in text.lower():
            return 1
    return None

# --- Step 4: Streamlit UI ---
st.set_page_config(page_title="Restaurant Review Predictor", layout="centered")
st.title("🍽️ Restaurant Review Rating Predictor")
st.markdown(
    "Enter a restaurant review below and get the predicted rating (1-5) with ⭐ and emojis!"
)

# Sidebar instructions
st.sidebar.header("How to Use")
st.sidebar.write(
    """
1. Enter your restaurant review text.
2. Click the **Predict Rating** button.
3. View the predicted rating with ⭐ and description with emojis.
"""
)

# User input
user_review = st.text_area("Enter Review Text", height=150)

if st.button("Predict Rating"):
    if user_review.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        # Preprocess the review
        cleaned_review = clean_text(user_review)

        # Check for negative keyword override
        override_rating = keyword_override(user_review)

        if override_rating is not None:
            prediction = override_rating
        else:
            review_vector = vectorizer.transform([cleaned_review])
            prediction = model.predict(review_vector)[0]

        # --- Step 5: Display predicted rating visually ---
        # Visual display: stars repeated by rating
        visual_symbols = "⭐" * prediction
        st.subheader("Predicted Rating")
        st.success(f"{visual_symbols} ({prediction}/5)")

        # Textual description with emoji
        rating_text = {
            1: "Very Poor 😡",
            2: "Poor 😞",
            3: "Average 😐",
            4: "Good 🙂",
            5: "Excellent 😍"
        }
        st.info(f"Rating Description: {rating_text[prediction]}")



