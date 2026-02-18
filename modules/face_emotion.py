import cv2
from fer import FER

# --------------------------------------------------
# LOAD MODEL ONCE (VERY IMPORTANT)
# --------------------------------------------------
# Loading inside function would freeze webcam
detector = FER(mtcnn=True)


# --------------------------------------------------
# FACE EMOTION DETECTION FUNCTION
# --------------------------------------------------
def detect_face_emotion(frame):
    """
    Takes a BGR frame (OpenCV image)
    Returns:
        emotion_label (str or None)
        confidence (float)
    """

    try:
        # FER expects RGB image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detector.detect_emotions(rgb_frame)

        if not results:
            return None, 0.0

        emotions = results[0]["emotions"]

        # Get dominant emotion
        emotion_label = max(emotions, key=emotions.get)
        confidence = float(emotions[emotion_label])

        return emotion_label, confidence

    except Exception:
        # Prevent webcam crash
        return None, 0.0
