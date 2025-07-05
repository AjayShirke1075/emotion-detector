import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load the model
@st.cache_resource
def load_emotion_model():
    return load_model("face_emotion_model.h5")

model = load_emotion_model()

# Emotion labels and emojis
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emoji_map = {
    "Happy": "😄", "Sad": "😢", "Angry": "😠",
    "Neutral": "😐", "Fear": "😨", "Disgust": "🤢", "Surprise": "😲"
}
suggestions_map = {
    "Happy": "Keep smiling and spread your positivity! 🌟",
    "Sad": "Try listening to your favorite song or talk to a friend. 💬",
    "Angry": "Take deep breaths and go for a short walk. 🧘",
    "Neutral": "Try something exciting to brighten your mood! ✨",
    "Fear": "Face your fears step by step — you're stronger than you think! 💪",
    "Disgust": "Clear your surroundings and take a break. 🧼",
    "Surprise": "Channel your surprise into curiosity. Learn something new! 📚"
}

# Page config
st.set_page_config(page_title="Emotion Detector", layout="wide")

# Sidebar
st.sidebar.title("🎛️ Control Panel")
run = st.sidebar.checkbox("📷 Start Webcam")
stop_btn = st.sidebar.checkbox("⏹️ Stop Webcam")

# Display area
FRAME_WINDOW = st.image([])
st.title("😊 Real-Time Emotion Detection with Suggestions")

# Store results
if 'emotion_data' not in st.session_state:
    st.session_state.emotion_data = []

# Run webcam
if run and not stop_btn:
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype('float') / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face)[0]
            max_idx = np.argmax(prediction)
            label = emotion_labels[max_idx]
            confidence = prediction[max_idx]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{label} ({confidence*100:.1f}%)"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            st.session_state.emotion_data.append((datetime.now(), label, confidence))

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

# After webcam stops
if stop_btn and st.session_state.emotion_data:
    st.success("✅ Webcam stopped. Displaying emotion analysis and suggestions...")

    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.emotion_data, columns=["Time", "Emotion", "Confidence"])

    # Emotion frequency chart
    st.subheader("📊 Emotion Frequency")
    freq = df['Emotion'].value_counts()
    st.bar_chart(freq)

    # Emotion over time
    st.subheader("📈 Emotion Over Time")
    df['Count'] = df.groupby('Emotion').cumcount()
    line_chart = df.pivot(index="Time", columns="Emotion", values="Count").fillna(0)
    st.line_chart(line_chart)

    # Latest emotion and suggestion
    last_emotion = df.iloc[-1]["Emotion"]
    last_conf = df.iloc[-1]["Confidence"]
    emoji = emoji_map.get(last_emotion, "🙂")
    suggestion = suggestions_map.get(last_emotion, "Stay positive and strong!")

    st.subheader(f"😊 Last Detected Emotion: {emoji} **{last_emotion}** ({last_conf*100:.2f}%)")
    st.info(f"💡 Suggestion: {suggestion}")

    # Export
    st.download_button("📥 Download Emotion Data", df.to_csv(index=False), "emotions.csv")

else:
    st.info("👈 Use the sidebar to start and stop webcam.")
