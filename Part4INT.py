import streamlit as st
import numpy as np
import cv2
import pathlib
import time
import threading
import logging

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cams_Res = []
coords = []
trajectory = []

pos_cam1 = None
pos_cam2 = None
lock = threading.Lock()
cams_res_lock = threading.Lock()
coords_lock = threading.Lock()
trajectory_lock = threading.Lock()

# HSV range for pink
lo = np.array([140, 50, 50])  # Lower bound (Hue, Saturation, Value)
hi = np.array([170, 255, 255])  # Upper bound (Hue, Saturation, Value)

def calculate_3d_position(uL, vL, uR, fx, fy, b, ox, oy):
    if uL == uR:
        raise ValueError("uL and uR cannot be the same value to avoid division by zero.")
    x = b * (uL - ox) / (uL - uR)
    y = b * fx * (vL - oy) / (fy * (uL - uR))
    z = b * fx / (uL - uR)
    return x, y, z

def detect_and_track(image, surface_min, surface_max, cam_id):
    points = []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        points.append([int(x), int(y), int(radius), int(area)])

    # Track trajectory
    if points:
        with trajectory_lock:
            if cam_id == 1:
                trajectory.append((cam_id, points[0][0], points[0][1]))
            elif cam_id == 2:
                trajectory.append((cam_id, points[0][0], points[0][1]))

    return hsv, mask, points

def draw_trajectory(frame):
    with trajectory_lock:
        for cam_id, x, y in trajectory:
            color = (0, 255, 0) if cam_id == 1 else (255, 0, 0)
            cv2.circle(frame, (x, y), 3, color, -1)


def object_tracking(name, http, cam_id, cam1_results, cam2_results):
    global pos_cam1, pos_cam2
    cap = cv2.VideoCapture(http)

    if not cap.isOpened():
        logging.error(f"Cannot open camera stream {http}")
        return

    output_width, output_height = 640, 480

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (output_width, output_height))
        hsv, mask, points = detect_and_track(frame, 200, 300, cam_id)

        with lock:
            if points:
                x, y, radius, area = points[0]
                cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)
                cv2.putText(frame, f"{x}, {y}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                if cam_id == 1:
                    pos_cam1 = (x, y)
                else:
                    pos_cam2 = (x, y)

                if pos_cam1 and pos_cam2:
                    ul, vl = pos_cam1
                    ur, vr = pos_cam2
                    fx = cam1_results[1][0, 0]
                    fy = cam1_results[1][1, 1]
                    ox = cam1_results[1][0, 2]
                    oy = cam1_results[1][1, 2]
                    b = distance_cameras

                    x1, y1, z = calculate_3d_position(ul, vl, ur, fx, fy, b, ox, oy)
                    cv2.putText(frame, f"{x1:.2f}, {y1:.2f}, {z:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        draw_trajectory(frame)

        cv2.imshow(f"{name}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

st.title("Enhanced Camera Calibration with Trajectory Tracking")

# Input form
with st.form("parameters"):
    distance_cameras = st.number_input("Distance between cameras (m)", min_value=0.0, value=1.0, step=0.1)
    nb_images = st.number_input("Calibration images", min_value=1, value=10, step=1)
    checkboard_size = st.text_input("Checkerboard size (e.g., '9x6')", value="9x6")
    address_camera1 = st.text_input("Camera 1 Address", value="http://...")
    address_camera2 = st.text_input("Camera 2 Address", value="http://...")
    submitted = st.form_submit_button("Submit")

if submitted:
    CHECKERBOARD = (int(checkboard_size.split("x")[0]), int(checkboard_size.split("x")[1]))

    # Launch calibration threads
    cam1_results = (True, np.eye(3), np.zeros(5), [], np.zeros((3, 1)))
    cam2_results = (True, np.eye(3), np.zeros(5), [], np.zeros((3, 1)))

    # Launch object tracking
    T1 = threading.Thread(target=object_tracking, args=("Camera 1", address_camera1, 1, cam1_results, cam2_results))
    T2 = threading.Thread(target=object_tracking, args=("Camera 2", address_camera2, 2, cam1_results, cam2_results))
    T1.start()
    T2.start()
    T1.join()
    T2.join()

    st.success("Trajectory tracking completed!")
