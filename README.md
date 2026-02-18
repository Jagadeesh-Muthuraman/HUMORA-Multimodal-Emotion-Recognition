# HUMORA – Multimodal Emotion Recognition System

HUMORA is a real-time AI-based multimodal emotion recognition system designed to detect human emotions using facial expressions, speech signals, and textual input. The system combines multiple modalities through decision-level fusion to produce a more reliable and robust emotional prediction.

## Overview

Traditional emotion recognition systems relying on a single modality often suffer from ambiguity and reduced accuracy. HUMORA addresses this limitation by integrating facial, audio, and textual emotional cues into a unified prediction framework.

## Features

- Real-time facial emotion detection using webcam input
- Audio emotion recognition from speech signals
- Text-based emotion analysis using Natural Language Processing
- Decision-level multimodal fusion
- Live analytics dashboard with emotion distribution and timeline visualization
- Automatic periodic emotion detection
- Session management with data reset functionality

## System Architecture

**Input Modalities**
- Face (Computer Vision)
- Audio (Speech Processing)
- Text (Natural Language Processing)

**Processing Pipeline**
- Feature extraction
- Individual emotion prediction models
- Decision-level fusion

**Output**
- Final emotion prediction
- Analytical visualization dashboard

## Technology Stack

- Python
- Streamlit
- OpenCV
- TensorFlow / Keras
- Librosa
- Scikit-learn
- streamlit-webrtc (WebRTC streaming)

## How to Run the Project

```bash
git clone https://github.com/Jagadeesh-Muthuraman/HUMORA-Multimodal-Emotion-Recognition.git
cd HUMORA-Multimodal-Emotion-Recognition
pip install -r requirements.txt
streamlit run app.py

Jagadeesh Muthuraman  
B.Tech Information Technology
