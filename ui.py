import streamlit as st
import cv2
import numpy as np
import os

# Load the Haar Cascade and recognizer
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer.yml")

# Load label mapping from folder names
def load_labels(dataset_path="known_faces"):
    label_map = {}
    for idx, name in enumerate(os.listdir(dataset_path)):
        label_map[idx] = name
    return label_map

labels = load_labels()

def detect_and_recognize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        label_id, confidence = recognizer.predict(roi)

        label = labels.get(label_id, "Unknown")
        confidence_text = f"{int(confidence)}"
        
        color = (0, 255, 0) if confidence < 70 else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} ({confidence_text})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return image

st.set_page_config(page_title="Face Recognizer", layout="centered")
st.title("🧠 Face Detection & Recognition App")
st.write("Upload a photo or use your webcam to detect faces and recognize people.")

option = st.radio("Select Mode", ["📷 Upload Image", "🎥 Live Webcam"])

if option == "📷 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        result_img = detect_and_recognize(img)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Detected", use_container_width=True)

elif option == "🎥 Live Webcam":
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam not available.")
            break

        frame = detect_and_recognize(frame)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    else:
        cap.release()
        cv2.destroyAllWindows()
