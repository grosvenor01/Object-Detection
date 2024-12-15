import torch
import cv2
import pandas as pd
import pathlib
from IPython.display import display

# Fix relative paths for Windows (only required if using Jupyter)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load model once at the start
def load_models():
    try:
        # Load model
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'last.pt', force_reload=True, trust_repo=True)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_models()

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

# Capture video stream
capture = cv2.VideoCapture("http://192.168.137.244:8080/video")
if not capture.isOpened():
    print("Error opening video stream.")
    exit()

# Process frames
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
