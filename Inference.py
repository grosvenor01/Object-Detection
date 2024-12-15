import torch
import cv2
import numpy as np
import pandas as pd
import pathlib
from IPython.display import display
#bonus code (usedfor the relatives path finding)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# models loading
def load_models():
    model = torch.hub.load('Ultralytics/yolov5', 'custom', "last.pt", force_reload=True, trust_repo=True)
    return model

# turning the result into a data frame with the predicted volume
def stacking_results(model_results):
    try:
        df = model_results.pandas().xyxy[0]
        return df
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        return pd.DataFrame() # Return an empty DataFrame on error

# build the boundings of a specific class
def boundings_builder(image ,  df):
    if image is None or df.empty:
        return image

    for _, row in df.iterrows():
        x1, y1, x2, y2 = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        #calculate center coordinates
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(image, (center_x, center_y), 5, (139, 69, 19), -1) # -1 fills the circle

    return image

# main code
model = load_models()
capture = cv2.VideoCapture("http://192.168.1.40:8080/video")

if not capture.isOpened():
    print("Error opening video stream.")
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        print("Error reading frame.")
        break

    results = model(frame)
    df = stacking_results(results)
    image = boundings_builder(frame, df)
    cv2.imshow('livestream', image)
    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()