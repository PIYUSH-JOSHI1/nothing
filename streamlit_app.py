import streamlit as st
import cv2
from cvzone.ClassificationModule import Classifier
from ultralytics import YOLO
import numpy as np
from PIL import Image

st.title("Live Object Detection using Webcam")

# Initialize the YOLO model
model = YOLO('yolov5s.pt')

def get_frame():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image")
            break
        yield frame
    cap.release()

def detect_objects(frame):
    results = model(frame)
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    n = len(labels)
    for i in range(n):
        row = cords[i]
        if row[4] >= 0.2:  # confidence threshold
            x1, y1, x2, y2 = int(row[0]*frame.shape[1]), int(row[1]*frame.shape[0]), int(row[2]*frame.shape[1]), int(row[3]*frame.shape[0])
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, f"{model.names[int(labels[i])]} {row[4]:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    return frame

frame_placeholder = st.empty()

for frame in get_frame():
    frame = detect_objects(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

