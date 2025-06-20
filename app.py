import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import time

# Load YOLO model
model = YOLO("runs/detect/train2/weights/best.pt")
class_names = model.names

st.title("🤟 ASL Detection with YOLOv8")
st.markdown("Upload an image or enable webcam to detect ASL signs.")

# Image uploader or webcam
option = st.radio("Choose input source:", ("Upload Image", "Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="Uploaded Image", channels="BGR")

        # Run detection
        results = model(image, imgsz=224, conf=0.5)[0]
        annotated = results.plot()

        st.image(annotated, caption="Predictions", channels="BGR")
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            st.success(f"Prediction: **{class_names[cls_id]}**, Confidence: **{conf:.2f}**")

elif option == "Webcam":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Webcam not accessible")
    else:
        run = st.checkbox("Start webcam")

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("No frame detected!")
                break

            results = model(frame, imgsz=224, conf=0.5)[0]
            annotated = results.plot()

            # Convert to RGB for Streamlit
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

            # Show prediction below
            if results.boxes:
                cls_id = int(results.boxes.cls[0])
                conf = float(results.boxes.conf[0])
                st.write(f"Prediction: **{class_names[cls_id]}**, Confidence: **{conf:.2f}**")

        cap.release()



