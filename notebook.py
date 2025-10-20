# notebook.py — Amazon Fine Food Reviews Rating Predictor
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

# Download NLTK data (stopwords & wordnet)
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean review text
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation & numbers
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]  # remove stopwords + lemmatize
    return " ".join(words)



# 1. Load dataset
data_path = "data/amazon_reviews.csv"
df = pd.read_csv(data_path)

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())

# 2. Clean & select useful columns
# The dataset columns: Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text
df = df[["Text", "Score"]].dropna()
df = df.rename(columns={"Text": "review_text", "Score": "rating"})
# Apply text cleaning to review_text
df['review_text'] = df['review_text'].apply(clean_text)


# 3. Reduce data (optional - makes it faster)
df = df.sample(10000, random_state=42)

# 4. Prepare labels
X = df["review_text"].values
y = df["rating"].astype(int).values

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. TF-IDF + Logistic Regression
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=300)
model.fit(X_train_tfidf, y_train)

# 7. Evaluate model
y_pred = model.predict(X_test_tfidf)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# 8. Save model & vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
joblib.dump(vectorizer, "models/vectorizer.joblib")

print("\n🎉 Model and vectorizer saved successfully in /models/")
