import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("data/text_emotion_dataset.csv")

X = data["text"]
y = data["emotion"]

# Convert text → numerical features
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_vec, y)

# Save model
pickle.dump(model, open("models/text_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/tfidf.pkl", "wb"))

print("✅ Text Emotion Model Trained Successfully!")
