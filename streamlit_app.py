import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.title("Live Object Detection using Webcam")

# Initialize the YOLO model with YOLOv8 weights
model = YOLO('yolov8l.pt')  # Make sure yolov8l.pt is available in the same directory or provide the correct path

def detect_objects(frame):
    results = model(frame)
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    n = len(labels)
    for i in range(n):
        row = cords[i]
        if row[4] >= 0.2:  # confidence threshold
            x1, y1, x2, y2 = int(row[0] * frame.shape[1]), int(row[1] * frame.shape[0]), int(row[2] * frame.shape[1]), int(row[3] * frame.shape[0])
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, f"{model.names[int(labels[i])]} {row[4]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    return frame

def get_frame():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image")
            break
        frame = detect_objects(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
    cap.release()

frame_placeholder = st.empty()

# Button to start webcam
if st.button("Start Webcam"):
    get_frame()

# File uploader to detect objects in an uploaded image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    detected_image = detect_objects(image)
    st.image(detected_image, channels="RGB", caption="Uploaded Image with Detected Objects")
