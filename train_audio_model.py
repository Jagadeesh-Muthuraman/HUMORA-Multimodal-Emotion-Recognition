import os
import librosa
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/audio"

emotion_map = {
    "01": "Neutral",
    "02": "Neutral",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprise"
}

X = []
y = []

def extract_features(file):
    audio, sr = librosa.load(file, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            emotion = emotion_map[emotion_code]

            file_path = os.path.join(root, file)

            features = extract_features(file_path)

            X.append(features)
            y.append(emotion)

print("Training model...")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Save both model AND scaler
pickle.dump(model, open("models/audio_model.pkl", "wb"))
pickle.dump(scaler, open("models/audio_scaler.pkl", "wb"))

print("✅ Audio Emotion Model Trained Successfully!")
