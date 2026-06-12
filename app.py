import cv2
import numpy as np
import os
import streamlit as st
from tensorflow.keras.models import load_model

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Real-time Emotion Detector", layout="centered")
st.title("🎬 Real-time Emotion Detector")
st.write("Detecting emotions using original Haar Cascade & Keras Model.")

# 💡 Set your default video file path here
default_video_path = 'sample video/sample_face_test.mp4'

# Streamlit Component: User Upload
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

video_target = None
if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    video_target = "temp_video.mp4"
else:
    if os.path.exists(default_video_path):
        video_target = default_video_path
    else:
        st.warning(f"⚠️ Default video not found at '{default_video_path}'. Please upload a video file.")

# --- LOAD MODEL & HAAR CASCADE ---
@st.cache_resource
def load_emotion_model():
    # 💡 Verify/adjust your actual model path here (e.g., 'model/DAN_E50.h5')
    model_path = 'model/DAN_E50.h5' 
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.error(f"❌ Model file '{model_path}' not found!")
        return None

@st.cache_resource
def load_cascade():
    # Load OpenCV's Haar Cascade to detect faces
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = load_emotion_model()
face_classifier = load_cascade()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- VIDEO PROCESSING ---
if video_target and model is not None and not face_classifier.empty():
    cap = cv2.VideoCapture(video_target)
    
    # 💡 Placeholder to display the video on Streamlit
    frame_placeholder = st.empty()
    stop_button = st.button("Stop Video Processing")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.info("🏁 Video playback finished.")
            break

        # 💡 [Following Original Code] Convert to Grayscale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 💡 [Following Original Code] Face Detection
        faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw a bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Crop the face area (ROI)
            roi_gray = gray_image[y:y+h, x:x+w]
            
            # Skip if an unexpected cropping error occurs
            if roi_gray.size == 0:
                continue

            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            if np.sum([roi_gray]) != 0:
                # 💡 [Following Original Code] Preprocess (divide by 255.0)
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)
                
                # Predict the emotion
                prediction = model.predict(roi, verbose=0)[0]
                label_index = prediction.argmax()
                label = emotion_labels[label_index]
                
                # Put the predicted emotion label on the frame
                label_position = (x, y - 10)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 💡 [Streamlit Modification] Convert OpenCV BGR to RGB and render on the web UI
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()