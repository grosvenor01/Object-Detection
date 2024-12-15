import streamlit as st
import numpy as np
import pandas as pd
import cv2
import numpy as np
import pathlib
import threading
import streamlit as st
import logging
import torch
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# --------------Color Detection section------------------
lo = np.array([140, 50, 50])  # Lower bound (Hue, Saturation, Value)
hi = np.array([170, 255, 255])  # Upper bound (Hue, Saturation, Value)

def detect_inrange(image):
    points = []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
    mask = cv2.inRange(blurred, lo, hi)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        points.append([int(x), int(y), int(radius), int(area)])
    return hsv, mask, points

def object_detection_color(name, http, num, points_history):
    cap = cv2.VideoCapture(http)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la caméra/flux vidéo {http}")
        return

    output_width = 640
    output_height = 480

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (output_width, output_height))
        hsv, mask, points = detect_inrange(frame)

        if points:
            x, y, radius, area = points[0]
            cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)
            cv2.putText(frame, f"{x}, {y}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            points_history[num-1].append((x,y)) #Append to the correct camera's history

        cv2.imshow(f"Image : {name}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --------------Yolo model section------------------
# Load model once at the start
def load_models():
    try:
        # Load model
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'last.pt', force_reload=True, trust_repo=True)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to convert model results to DataFrame
def stacking_results(model_results):
    return model_results.pandas().xyxy[0] if model_results else pd.DataFrame()

# Function to draw bounding boxes and centers
def boundings_builder(image, df):
    if image is None or df.empty:
        return image

    for _, row in df.iterrows():
        # Get coordinates and calculate center
        x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw center circle (consider changing the circle radius if needed)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image

def Yolo_detection(address):
    model = load_models()
    capture = cv2.VideoCapture(address)
    if not capture.isOpened():
        print("Error opening video stream.")
        exit()
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error reading frame.")
            break
        # Object detection using YOLOv5 model
        results = model(frame)  # Use the pre-loaded model
        df = stacking_results(results)  # Convert results to a DataFrame
        frame_with_bounding = boundings_builder(frame, df)  # Draw bounding boxes and centers
        # Resize image if necessary (to fit within a small window)
        frame_resized = cv2.resize(frame_with_bounding, (800, 450))
        # Display the processed frame
        cv2.imshow('Livestream', frame_resized)
        # Exit on pressing 'q'
        if cv2.waitKey(1) == ord("q"):
            break
    # Release resources
    capture.release()
    cv2.destroyAllWindows()

st.title("Object Detection")
with st.form("parameters"):
    address_camera1 = st.text_input("Adresse caméra 1 ", value="http://...")
    models_to_use = st.selectbox("choose the model", ["Color detection" , "YOLO"])
    submitted = st.form_submit_button("Valider")

if submitted:
    if models_to_use == "Color detection":
        points_history = [[], []] #Initialize history for two cameras
        T1 = threading.Thread(target=object_detection_color, args=("Camera 1", address_camera1, 1, points_history))
        T1.start()
        T1.join()

        cv2.destroyAllWindows()

    elif models_to_use =="YOLO":
        T1 = threading.Thread(target=Yolo_detection, args=(address_camera1,))
        T1.start()
        T1.join()
