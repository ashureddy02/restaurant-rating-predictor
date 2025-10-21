import streamlit as st
import joblib
import re

# Load model & vectorizer
model = joblib.load("models/model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

# --- TEXT CLEANING FUNCTION ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- EXPANDED NEGATIVE KEYWORDS ---
negative_keywords = [
    "bad", "terrible", "worst", "disappoint", "slow", "rude", "cold", "bland",
    "small", "stale", "overcooked", "under", "delay", "expensive",
    "unfriendly", "inattentive", "dirty", "poor", "greasy", "burnt",
    "waited", "mistake", "forgot", "but", "waste", "noisy",
    "soggy", "overpriced", "underwhelming", "smelly", "mediocre",
    "unpleasant", "dry", "unclean", "unhygienic", "late",
    "crowded", "chaotic", "inconsistent", "lacking", "blurry"
]

# --- POSITIVE KEYWORDS ---
positive_keywords = [
    "good", "great", "amazing", "excellent", "fantastic", "tasty",
    "fresh", "delicious", "friendly", "cozy", "beautiful", "crisp",
    "crispy", "wellcooked", "flavourful", "spiced", "perfect", "love",
    "awesome", "pleasant", "fast", "clean", "worth", "refreshing"
]

# --- MODERATE KEYWORDS ---
moderate_keywords = [
    "okay", "average", "fine", "nothing", "decent", "normal", "mediocre",
    "alright", "not bad", "could be better"
]

# --- KEYWORD OVERRIDE FOR STRONG NEGATIVES ---
def keyword_override(text):
    return None  # handled later with 2+ negative keywords

# --- PREDICTION FUNCTION ---
def predict_rating(review_text):
    if not review_text.strip():
        return 2, "Please enter a review to get a prediction!! 🛑"

    text_cleaned = clean_text(review_text)

    # --- Safety check for meaningless / random / very short input ---
    keywords_for_review = r'\b(food|taste|service|restaurant|dish|staff|ambience|price|menu|waiter|clean|delicious|meal)\b'
    if len(review_text.split()) < 4 or re.match(r'^[\d\s\W]+$', review_text) \
       or not re.search(keywords_for_review, review_text.lower()):
        prediction = 2
        message = "Thanks! Please add more details about the food or service 🍽️"
        return prediction, message

    # --- Count keyword hits ---
    neg_hits = sum(kw in text_cleaned for kw in negative_keywords)
    pos_hits = sum(kw in text_cleaned for kw in positive_keywords)
    mid_hits = sum(kw in text_cleaned for kw in moderate_keywords)

    # --- Strong negative override (only if 2+ negative keywords) ---
    if neg_hits >= 2:
        return 1, None

    # --- ML prediction ---
    review_vector = vectorizer.transform([text_cleaned])
    prediction = model.predict(review_vector)[0]

    # --- 5-STAR LOGIC ---
    if pos_hits >= 3:
        prediction = 5  # Strong positive
    elif pos_hits > 0 and (neg_hits > 0 or mid_hits > 0):
        prediction = 2  # Slightly negative / improvement feedback
    elif mid_hits > 0 or (neg_hits > 0 and pos_hits > 0):
        prediction = 3  # Mixed / moderate
    elif pos_hits > neg_hits:
        prediction = 4  # Mostly positive / minor negatives
    else:
        prediction = 2  # Meaningless / very short / random input

    return prediction, None

# --- Streamlit UI ---
st.set_page_config(page_title="Restaurant Rating Predictor", page_icon="🍽️")

st.title("🍽️ Restaurant Review Rating Predictor")
st.markdown("### Enter a restaurant review and see the predicted rating!")

# --- Sidebar instructions ---
st.sidebar.header("How to Use")
st.sidebar.write(
    """
1. Enter your restaurant review text.
2. Click the **Predict Rating** button.
3. See your predicted rating and explanation.
"""
)

review_text = st.text_area("Write your restaurant review below:", height=150)

if st.button("Predict Rating"):
    rating, message = predict_rating(review_text)

    if message:
        st.warning(message)
    else:
        stars = "⭐" * rating + f" ({rating}/5)"
        st.subheader("Predicted Rating")
        st.write(stars)

        descriptions = {
            1: "Very Poor 😠",
            2: "Needs Improvement 😕",
            3: "Average 🙂",
            4: "Good 👍",
            5: "Excellent 😍"
        }
        st.write(f"**Rating Description:** {descriptions.get(rating, 'Unknown')}")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit & Machine Learning")

