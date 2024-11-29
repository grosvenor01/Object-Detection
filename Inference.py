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
def boundings_builder(image, df):
    if image is None or df.empty:
        return image

    for _, row in df.iterrows():
        x1, y1, x2, y2, label = row["xmin"], row["ymin"], row["xmax"], row["ymax"], row["class"]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (139, 69, 19), 2)
        cv2.putText(image, str(label), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 69, 19), 2, cv2.LINE_AA)

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