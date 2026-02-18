import os
import pandas as pd
import streamlit as st
from datetime import datetime

FILE_PATH = "emotion_history.csv"


# ---------------------------------------------------
# SAVE EMOTION
# ---------------------------------------------------
def save_emotion(emotion):

    timestamp = datetime.now()

    new_row = pd.DataFrame(
        [[timestamp, emotion]],
        columns=["timestamp", "emotion"]
    )

    if os.path.exists(FILE_PATH):
        df = pd.read_csv(FILE_PATH)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(FILE_PATH, index=False)


# ---------------------------------------------------
# LOAD HISTORY
# ---------------------------------------------------
def load_history():

    if os.path.exists(FILE_PATH):
        return pd.read_csv(FILE_PATH)

    return pd.DataFrame(columns=["timestamp", "emotion"])


# ---------------------------------------------------
# CLEAR HISTORY (FULL RESET)
# ---------------------------------------------------
def clear_history():

    # Delete CSV file
    if os.path.exists(FILE_PATH):
        os.remove(FILE_PATH)

    # Remove ALL emotion-related session states
    keys_to_remove = [
        "live_face_result",
        "last_auto_run",
        "confirm_clear",
    ]

    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
