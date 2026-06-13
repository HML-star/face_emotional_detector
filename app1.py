import cv2
import numpy as np
import os
import mediapipe as mp
import streamlit as st  # Fixed: Changed from 'as None' to 'as st' to avoid NameError
from tensorflow.keras.models import load_model

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Real-time Emotion Detector", layout="centered")
st.title("🎬 Real-time Emotion Detector")
st.write("Upload a video or use the default one to detect emotions using MediaPipe & Keras.")

# 💡 Set your default video file path here
default_video_path = 'sample video/Face_Test2.mp4'

# Streamlit Component: Allows users to upload their own video file directly
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

video_target = None
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    video_target = "temp_video.mp4"
else:
    # If no file is uploaded, fall back to the provided default video
    if os.path.exists(default_video_path):
        video_target = default_video_path
    else:
        st.warning(f"⚠️ Default video not found at '{default_video_path}'. Please upload a video file.")

# --- MODEL & MEDIAPIPE INITIALIZATION ---
@st.cache_resource  # Cache the model to prevent reloading on every rerun
def load_emotion_model():
    # Verify your model path here
    if os.path.exists('model/DAN_E50.h5'):
        return load_model('model/DAN_E50.h5')
    else:
        st.error("❌ Model file 'model/DAN_E50.h5' not found!")
        return None

model = load_emotion_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# --- VIDEO PROCESSING ---
if video_target and model is not None:
    cap = cv2.VideoCapture(video_target)
    
    # 💡 Create an empty placeholder to dynamically display the video frames in Streamlit
    frame_placeholder = st.empty()
    
    # Stop Button (used to interrupt and stop the video processing)
    stop_button = st.button("Stop Video Processing")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.info("🏁 Video playback finished.")
            break

        h_frame, w_frame, _ = frame.shape
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w_frame)
                y = int(bboxC.ymin * h_frame)
                w = int(bboxC.width * w_frame)
                h = int(bboxC.height * h_frame)

                x, y = max(0, x), max(0, y)
                w, h = max(0, w), max(0, h)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Preprocessing for Emotion Model
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_gray = gray_image[y:y+h, x:x+w]
                
                if roi_gray.size != 0:
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                    
                    if np.sum([roi_gray]) != 0:
                        roi = roi_gray.astype('float') / 255.0
                        roi = np.expand_dims(roi, axis=0)
                        roi = np.expand_dims(roi, axis=-1)
                        
                        # Predict
                        prediction = model.predict(roi, verbose=0)[0]
                        label_index = prediction.argmax()
                        label = emotion_labels[label_index]
                        
                        # Write Text
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 💡 CRITICAL CHANGE: Convert OpenCV BGR image to RGB and render it directly on the Streamlit interface
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    face_detection.close()
    
# --- IMAGE EMOTION DETECTION ---
st.subheader("🖼️ Image Emotion Detection")

uploaded_image = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    key="image_uploader"
)

if uploaded_image is not None and model is not None:

    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb_image)

    if results.detections:
        h_img, w_img, _ = image.shape

        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box

            x = int(bboxC.xmin * w_img)
            y = int(bboxC.ymin * h_img)
            w = int(bboxC.width * w_img)
            h = int(bboxC.height * h_img)

            x, y = max(0, x), max(0, y)
            w, h = max(0, w), max(0, h)

            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[y:y+h, x:x+w]

            if roi_gray.size != 0:
                roi_gray = cv2.resize(roi_gray, (48, 48))

                roi = roi_gray.astype("float") / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)

                prediction = model.predict(roi, verbose=0)[0]
                label = emotion_labels[np.argmax(prediction)]

                cv2.putText(
                    image,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            channels="RGB",
            caption="Detected Emotion"
        )

    else:
        st.warning("No face detected in the image.")