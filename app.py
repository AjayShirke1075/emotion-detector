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
emotion_colors = {
    "Happy": "#00ff88", "Sad": "#4a9eff", "Angry": "#ff4757",
    "Neutral": "#a29bfe", "Fear": "#fd79a8", "Disgust": "#fdcb6e", "Surprise": "#00cec9"
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
st.set_page_config(page_title="🧠 AI Emotion Detector", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Dark Tech Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 50%, #16213e 100%);
        background-attachment: fixed;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Animated Background Pattern */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(0, 255, 136, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(74, 158, 255, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(255, 71, 87, 0.05) 0%, transparent 50%);
        animation: pulse 8s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        background: linear-gradient(135deg, #00ff88, #00d4ff, #a29bfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: textGlow 3s ease-in-out infinite;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
    }
    
    @keyframes textGlow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.3); }
    }
    
    p, div, span, label {
        font-family: 'Rajdhani', sans-serif !important;
        color: #e0e0e0 !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(26, 26, 46, 0.95) 0%, rgba(10, 14, 39, 0.95) 100%);
        backdrop-filter: blur(10px);
        border-right: 2px solid rgba(0, 255, 136, 0.2);
        box-shadow: 5px 0 30px rgba(0, 255, 136, 0.1);
    }
    
    [data-testid="stSidebar"] h1 {
        font-size: 1.8rem !important;
        text-align: center;
        padding: 20px 0;
        animation: slideInLeft 0.8s ease-out;
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Checkbox Styling */
    .stCheckbox {
        background: rgba(255, 255, 255, 0.03);
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border: 1px solid rgba(0, 255, 136, 0.2);
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }
    
    .stCheckbox:hover {
        background: rgba(0, 255, 136, 0.1);
        border-color: rgba(0, 255, 136, 0.5);
        transform: translateX(5px);
        box-shadow: 0 5px 20px rgba(0, 255, 136, 0.2);
    }
    
    /* Main Title */
    .main-title {
        text-align: center;
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        margin: 30px 0 !important;
        animation: fadeInDown 1s ease-out;
        position: relative;
    }
    
    @keyframes fadeInDown {
        from { transform: translateY(-30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 20px 0;
        animation: fadeInUp 0.8s ease-out;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 255, 136, 0.2);
        border-color: rgba(0, 255, 136, 0.3);
    }
    
    @keyframes fadeInUp {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* Webcam Frame */
    [data-testid="stImage"] {
        border-radius: 20px;
        overflow: hidden;
        border: 3px solid rgba(0, 255, 136, 0.3);
        box-shadow: 0 0 40px rgba(0, 255, 136, 0.3);
        animation: borderPulse 2s ease-in-out infinite;
    }
    
    @keyframes borderPulse {
        0%, 100% { border-color: rgba(0, 255, 136, 0.3); box-shadow: 0 0 40px rgba(0, 255, 136, 0.3); }
        50% { border-color: rgba(0, 212, 255, 0.5); box-shadow: 0 0 60px rgba(0, 212, 255, 0.5); }
    }
    
    /* Info/Success Boxes */
    .stAlert {
        background: rgba(0, 255, 136, 0.1) !important;
        border: 1px solid rgba(0, 255, 136, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
        animation: slideInRight 0.6s ease-out;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Charts */
    [data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(0, 255, 136, 0.2);
        backdrop-filter: blur(5px);
    }
    
    /* Subheaders */
    .stMarkdown h2, .stMarkdown h3 {
        margin-top: 30px !important;
        padding: 15px 0 !important;
        border-bottom: 2px solid rgba(0, 255, 136, 0.2);
        animation: fadeIn 1s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Download Button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #00ff88, #00d4ff) !important;
        color: #0a0e27 !important;
        font-weight: 700 !important;
        font-family: 'Orbitron', sans-serif !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 30px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 20px rgba(0, 255, 136, 0.3) !important;
    }
    
    .stDownloadButton button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 8px 30px rgba(0, 255, 136, 0.5) !important;
    }
    
    /* Radio Buttons */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.03);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid rgba(0, 255, 136, 0.2);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        padding: 20px;
        border-radius: 15px;
        border: 2px dashed rgba(0, 255, 136, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0, 255, 136, 0.6);
        background: rgba(0, 255, 136, 0.05);
    }
    
    /* Regular Buttons */
    .stButton button {
        background: linear-gradient(135deg, #00ff88, #00d4ff) !important;
        color: #0a0e27 !important;
        font-weight: 700 !important;
        font-family: 'Orbitron', sans-serif !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 30px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 20px rgba(0, 255, 136, 0.3) !important;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 8px 30px rgba(0, 255, 136, 0.5) !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00ff88, #00d4ff) !important;
    }
    
    /* Emotion Card */
    .emotion-card {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 212, 255, 0.1));
        border: 2px solid rgba(0, 255, 136, 0.3);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 40px rgba(0, 255, 136, 0.2);
        animation: scaleIn 0.5s ease-out;
    }
    
    @keyframes scaleIn {
        from { transform: scale(0.9); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    
    .emotion-emoji {
        font-size: 5rem;
        animation: bounce 2s ease-in-out infinite;
        display: inline-block;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Loading Animation */
    .loading-spinner {
        border: 4px solid rgba(0, 255, 136, 0.1);
        border-top: 4px solid #00ff88;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Confidence Bar */
    .confidence-bar {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        height: 30px;
        overflow: hidden;
        margin: 15px 0;
        border: 1px solid rgba(0, 255, 136, 0.2);
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #00ff88, #00d4ff);
        border-radius: 10px;
        animation: fillBar 1s ease-out;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    
    @keyframes fillBar {
        from { width: 0; }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00ff88, #00d4ff);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #00d4ff, #a29bfe);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("# 🎛️ Control Panel")
st.sidebar.markdown("---")

# Mode Selection
mode = st.sidebar.radio(
    "Select Mode",
    ["📷 Webcam", "🎬 Video Upload"],
    help="Choose between live webcam or video file analysis"
)

st.sidebar.markdown("---")

if mode == "📷 Webcam":
    run = st.sidebar.checkbox("📷 Start Webcam", help="Activate your webcam to detect emotions in real-time")
    stop_btn = st.sidebar.checkbox("⏹️ Stop Webcam", help="Stop the webcam and view analysis")
    
    if run and not stop_btn:
        st.sidebar.markdown("### 🔴 LIVE")
        st.sidebar.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown("### ⚪ Standby")
else:
    run = False
    stop_btn = False
    st.sidebar.markdown("### 📤 Upload Video")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file for emotion analysis"
    )
    
    if uploaded_file is not None:
        process_video = st.sidebar.button("🚀 Process Video", help="Start analyzing the uploaded video")
    else:
        process_video = False

# Main Title
st.markdown('<h1 class="main-title">🧠 AI Emotion Detector</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; margin-bottom: 40px;">Real-Time Facial Emotion Recognition with AI-Powered Insights</p>', unsafe_allow_html=True)

# Display area
FRAME_WINDOW = st.empty()

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

            prediction = model.predict(face, verbose=0)[0]
            max_idx = np.argmax(prediction)
            label = emotion_labels[max_idx]
            confidence = prediction[max_idx]

            # Enhanced visualization with neon colors
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 136), 3)
            text = f"{label} ({confidence*100:.1f}%)"
            
            # Add background to text for better visibility
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x, y-35), (x+text_width, y), (0, 0, 0), -1)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 136), 2)

            st.session_state.emotion_data.append((datetime.now(), label, confidence))

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    cap.release()

# Video Upload Processing
elif mode == "🎬 Video Upload" and uploaded_file is not None and process_video:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🎬 Processing Video...")
    
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    
    # Process video
    cap = cv2.VideoCapture(tfile.name)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_display = st.empty()
    
    frame_count = 0
    video_emotions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 5th frame for efficiency
        if frame_count % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                face = face.astype('float') / 255.0
                face = np.expand_dims(face, axis=-1)
                face = np.expand_dims(face, axis=0)
                
                prediction = model.predict(face, verbose=0)[0]
                max_idx = np.argmax(prediction)
                label = emotion_labels[max_idx]
                confidence = prediction[max_idx]
                
                # Enhanced visualization
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 136), 3)
                text = f"{label} ({confidence*100:.1f}%)"
                
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(frame, (x, y-35), (x+text_width, y), (0, 0, 0), -1)
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 136), 2)
                
                timestamp = frame_count / fps
                video_emotions.append((timestamp, label, confidence))
            
            # Update display
            frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing: {frame_count}/{total_frames} frames ({progress*100:.1f}%)")
    
    cap.release()
    progress_bar.progress(1.0)
    status_text.text("✅ Video processing complete!")
    
    # Store results
    st.session_state.emotion_data = [(datetime.now(), e[1], e[2]) for e in video_emotions]
    st.session_state.video_emotions = video_emotions
    stop_btn = True  # Trigger results display
    
    st.markdown('</div>', unsafe_allow_html=True)

# After webcam stops or video processing
if (stop_btn or (mode == "🎬 Video Upload" and 'emotion_data' in st.session_state)) and st.session_state.emotion_data:
    if mode == "📷 Webcam":
        st.success("✅ Webcam session completed! Displaying comprehensive emotion analysis...")
    else:
        st.success("✅ Video analysis complete! Displaying comprehensive emotion insights...")
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.emotion_data, columns=["Time", "Emotion", "Confidence"])

    # Latest emotion and suggestion
    last_emotion = df.iloc[-1]["Emotion"]
    last_conf = df.iloc[-1]["Confidence"]
    emoji = emoji_map.get(last_emotion, "🙂")
    suggestion = suggestions_map.get(last_emotion, "Stay positive and strong!")
    emotion_color = emotion_colors.get(last_emotion, "#00ff88")

    # Emotion Card
    st.markdown(f"""
    <div class="emotion-card">
        <div class="emotion-emoji">{emoji}</div>
        <h2 style="margin: 20px 0; font-size: 2.5rem;">Detected: {last_emotion}</h2>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {last_conf*100}%;"></div>
        </div>
        <p style="font-size: 1.3rem; margin-top: 10px;">Confidence: {last_conf*100:.2f}%</p>
        <div style="margin-top: 30px; padding: 20px; background: rgba(0, 255, 136, 0.05); border-radius: 15px; border: 1px solid rgba(0, 255, 136, 0.2);">
            <p style="font-size: 1.2rem; margin: 0;">💡 <strong>Suggestion:</strong> {suggestion}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Create two columns for charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Emotion Distribution")
        freq = df['Emotion'].value_counts()
        st.bar_chart(freq, color="#00ff88")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📈 Emotion Timeline")
        df['Count'] = df.groupby('Emotion').cumcount()
        line_chart = df.pivot(index="Time", columns="Emotion", values="Count").fillna(0)
        st.line_chart(line_chart)
        st.markdown('</div>', unsafe_allow_html=True)

    # Statistics
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📋 Professional Analytics Dashboard")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Total Detections", len(df), delta=None)
    
    with stat_col2:
        dominant_emotion = df['Emotion'].mode()[0]
        dominant_pct = (df['Emotion'] == dominant_emotion).sum() / len(df) * 100
        st.metric("Dominant Emotion", dominant_emotion, delta=f"{dominant_pct:.1f}%")
    
    with stat_col3:
        avg_confidence = df['Confidence'].mean() * 100
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%", delta=None)
    
    with stat_col4:
        unique_emotions = df['Emotion'].nunique()
        st.metric("Unique Emotions", unique_emotions, delta=None)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced Analytics
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔍 Detailed Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("#### 📊 Emotion Breakdown")
        emotion_pcts = df['Emotion'].value_counts(normalize=True) * 100
        for emotion, pct in emotion_pcts.items():
            emoji = emoji_map.get(emotion, "🙂")
            color = emotion_colors.get(emotion, "#00ff88")
            st.markdown(f"""
            <div style="padding: 10px; margin: 5px 0; background: rgba(255,255,255,0.03); border-radius: 10px; border-left: 4px solid {color};">
                <span style="font-size: 1.2rem;">{emoji} <strong>{emotion}</strong></span>
                <span style="float: right; font-size: 1.1rem;">{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
    
    with insight_col2:
        st.markdown("#### 🎯 Peak Emotions")
        # Find highest confidence for each emotion
        peak_emotions = df.groupby('Emotion')['Confidence'].max().sort_values(ascending=False)
        for emotion, conf in peak_emotions.head(5).items():
            emoji = emoji_map.get(emotion, "🙂")
            color = emotion_colors.get(emotion, "#00ff88")
            st.markdown(f"""
            <div style="padding: 10px; margin: 5px 0; background: rgba(255,255,255,0.03); border-radius: 10px; border-left: 4px solid {color};">
                <span style="font-size: 1.2rem;">{emoji} <strong>{emotion}</strong></span>
                <span style="float: right; font-size: 1.1rem;">{conf*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Emotion Transitions
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔄 Emotion Transitions")
    
    if len(df) > 1:
        transitions = []
        for i in range(len(df) - 1):
            from_emotion = df.iloc[i]['Emotion']
            to_emotion = df.iloc[i + 1]['Emotion']
            if from_emotion != to_emotion:
                transitions.append(f"{from_emotion} → {to_emotion}")
        
        if transitions:
            transition_counts = pd.Series(transitions).value_counts().head(5)
            st.markdown("**Top 5 Emotion Shifts:**")
            for transition, count in transition_counts.items():
                st.markdown(f"- {transition}: **{count}** times")
        else:
            st.info("No emotion transitions detected - stable emotion throughout session")
    else:
        st.info("Need more data points to analyze transitions")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Confidence Trends
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📈 Confidence Trends Over Time")
    
    df_plot = df.copy()
    df_plot['Confidence_Pct'] = df_plot['Confidence'] * 100
    st.line_chart(df_plot.set_index('Time')['Confidence_Pct'], color="#00ff88")
    
    min_conf = df['Confidence'].min() * 100
    max_conf = df['Confidence'].max() * 100
    
    conf_col1, conf_col2 = st.columns(2)
    with conf_col1:
        st.metric("Minimum Confidence", f"{min_conf:.1f}%")
    with conf_col2:
        st.metric("Maximum Confidence", f"{max_conf:.1f}%")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Export
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        "📥 Download Emotion Data",
        df.to_csv(index=False),
        "emotions.csv",
        "text/csv",
        help="Download your emotion data as CSV"
    )

else:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 60px;">
        <h2>👈 Get Started</h2>
        <p style="font-size: 1.3rem; margin-top: 20px;">Use the sidebar to start your webcam and begin detecting emotions in real-time.</p>
        <p style="font-size: 1.1rem; margin-top: 15px; opacity: 0.8;">Our AI will analyze your facial expressions and provide personalized insights.</p>
    </div>
    """, unsafe_allow_html=True)
