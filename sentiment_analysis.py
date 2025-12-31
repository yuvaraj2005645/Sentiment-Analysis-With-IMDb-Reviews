# Sentiment Analysis with IMDb Reviews


import nltk
import pandas as pd
import re

from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download dataset (only first time)
nltk.download('movie_reviews')

# -----------------------------
# Load IMDb Reviews
# -----------------------------
documents = []
labels = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        review = movie_reviews.raw(fileid)
        documents.append(review)
        labels.append(category)

# Convert to DataFrame
df = pd.DataFrame({
    'review': documents,
    'sentiment': labels
})

# Convert labels to numbers
# pos -> 1 , neg -> 0
df['sentiment'] = df['sentiment'].map({'pos': 1, 'neg': 0})

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['review'] = df['review'].apply(clean_text)

# -----------------------------
# Split Data
# -----------------------------
X = df['review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Convert Text to Numbers (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# -----------------------------
# Evaluate Model
# -----------------------------
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# Test with Custom Review
# -----------------------------
def predict_sentiment(text):
    text = clean_text(text)
    text_vec = vectorizer.transform([text])
    result = model.predict(text_vec)[0]
    return "Positive 😊" if result == 1 else "Negative 😡"

# Example
review = "The movie was fantastic with great acting"
print("\nReview:", review)
print("Predicted Sentiment:", predict_sentiment(review))
