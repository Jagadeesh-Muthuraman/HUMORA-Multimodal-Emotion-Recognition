import librosa
import numpy as np
import pickle

# ---------------------------------
# LOAD TRAINED MODEL + SCALER
# ---------------------------------
model = pickle.load(open("models/audio_model.pkl", "rb"))
scaler = pickle.load(open("models/audio_scaler.pkl", "rb"))

# ---------------------------------
# FEATURE EXTRACTION FUNCTION
# ---------------------------------
def extract_features(file_path):

    # Load audio (limit to 3 seconds)
    audio, sr = librosa.load(file_path, duration=3)

    # Ensure fixed length (VERY IMPORTANT)
    target_length = sr * 3

    if len(audio) < target_length:
        audio = librosa.util.pad_center(audio, size=target_length)
    else:
        audio = audio[:target_length]

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    # Take mean across time axis
    features = np.mean(mfcc.T, axis=0)

    return features


# ---------------------------------
# PREDICTION FUNCTION
# ---------------------------------
def predict_audio_emotion(file_path):

    features = extract_features(file_path)

    # reshape for sklearn
    features = features.reshape(1, -1)

    # apply same scaling used during training
    features = scaler.transform(features)

    prediction = model.predict(features)[0]
    probability = max(model.predict_proba(features)[0])

    # ---------------------------------
    # EMOTION SMOOTHING (dataset bias fix)
    # ---------------------------------
    emotion_mapping = {
        "Disgust": "Sad",   # common MFCC confusion
        "Calm": "Neutral"
    }

    prediction = emotion_mapping.get(prediction, prediction)

    return {
        "label": prediction,
        "score": float(probability)
    }
