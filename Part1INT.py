import streamlit as st
import numpy as np
import cv2
import numpy as np
import pathlib
import threading
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


lo = np.array([140, 50, 50])  # Lower bound (Hue, Saturation, Value)
hi = np.array([170, 255, 255])  # Upper bound (Hue, Saturation, Value)

def detect_inrange(image, surface_min, surface_max):
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

def object_detection(name, http, num, points_history):
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
        hsv, mask, points = detect_inrange(frame, 200, 300)

        if points:
            x, y, radius, area = points[0]
            cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)
            cv2.putText(frame, f"{x}, {y}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            points_history[num-1].append((x,y)) #Append to the correct camera's history

            #Draw the path
            for i in range(1,len(points_history[num-1])):
                cv2.line(frame, points_history[num-1][i-1], points_history[num-1][i], (0,255,0), 2)


        cv2.imshow(f"Image : {name}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

st.title("Calibration Caméras")
with st.form("parameters"):
    address_camera1 = st.text_input("Adresse caméra 1 ", value="http://...")
    address_camera2 = st.text_input("Adresse caméra 2 ", value="http://...")
    submitted = st.form_submit_button("Valider")

if submitted:
    points_history = [[], []] #Initialize history for two cameras
    T1 = threading.Thread(target=object_detection, args=("Camera 1", address_camera1, 1, points_history))
    T2 = threading.Thread(target=object_detection, args=("Camera 2", address_camera2, 2, points_history))
    T1.start()
    T2.start()
    T1.join()
    T2.join()
    cv2.destroyAllWindows()