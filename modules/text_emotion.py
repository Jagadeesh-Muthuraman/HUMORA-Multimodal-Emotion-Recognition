import pickle

# Load trained model and vectorizer
model = pickle.load(open("models/text_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf.pkl", "rb"))

def predict_text_emotion(text):

    vec = vectorizer.transform([text])

    prediction = model.predict(vec)[0]
    probability = max(model.predict_proba(vec)[0])

    # Emotion mapping
    mapping = {
        "joy": "Happy",
        "sadness": "Sad",
        "anger": "Angry",
        "fear": "Fear",
        "surprise": "Surprise",
        "love": "Happy"
    }

    prediction = mapping.get(prediction, prediction)

    return {
        "label": prediction,
        "score": float(probability)
    }

