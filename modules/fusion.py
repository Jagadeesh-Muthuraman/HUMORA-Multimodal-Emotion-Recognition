# ---------------------------------
# MULTIMODAL FUSION ENGINE
# ---------------------------------

def fuse_emotions(text_result=None, audio_result=None, face_result=None):
    """
    Decision-level fusion using weighted confidence scoring.
    """

    # modality reliability weights
    weights = {
        "text": 0.5,   # semantic meaning (most reliable)
        "audio": 0.2,  # tone-based
        "face": 0.3    # visual emotion
    }

    emotion_scores = {}

    # ---------- TEXT ----------
    if text_result is not None:
        emotion = text_result["label"]
        score = text_result["score"] * weights["text"]

        emotion_scores[emotion] = emotion_scores.get(emotion, 0) + score

    # ---------- AUDIO ----------
    if audio_result is not None:
        emotion = audio_result["label"]
        score = audio_result["score"] * weights["audio"]

        emotion_scores[emotion] = emotion_scores.get(emotion, 0) + score

    # ---------- FACE ----------
    if face_result is not None:
        emotion = face_result["label"]
        score = face_result["score"] * weights["face"]

        emotion_scores[emotion] = emotion_scores.get(emotion, 0) + score

    # ---------- FINAL DECISION ----------
    if len(emotion_scores) == 0:
        return None, {}

    final_emotion = max(emotion_scores, key=emotion_scores.get)

    return final_emotion, emotion_scores
