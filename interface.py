import streamlit as st
import numpy as np
import torch
import cv2
import numpy as np
import pandas as pd
import pathlib
import time
import threading
from PIL import Image
import asyncio
import streamlit as st
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
cams_Res=[]
coords = []
cams_res_lock = threading.Lock()
coords_lock = threading.Lock()
global center_x, center_y
# HSV range for pink (adjust these values if needed)
lo = np.array([140, 50, 50])  # Lower bound (Hue, Saturation, Value)
hi = np.array([170, 255, 255])  # Upper bound (Hue, Saturation, Value)
# models loading
import torch
import logging


def calculate_3d_position(uL, vL, uR, fx, fy, b, ox, oy):
    if uL == uR:
        raise ValueError("uL and uR cannot be the same value to avoid division by zero.")
    x = b * (uL - ox) / (uL - uR)
    y = b * fx * (vL - oy) / (fy * (uL - uR))
    z = b * fx / (uL - uR)
    return x, y, z

def calibrate_camera(name , http , CHECKERBOARD ,  iteration ):
    # Criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Vectors to store 3D (real-world) and 2D (image) points
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    # Real-world 3D coordinates for the checkerboard
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # Calibration from the video 
    cap = cv2.VideoCapture(http)

    if not cap.isOpened():
        print('Error opening video capture')
        exit(0)

    # Timing to control frame capture
    last_capture_time = time.time()
    captured_images = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        # Get the current time
        current_time = time.time()

        # Capture a frame every 3 seconds
        if current_time - last_capture_time >= 1:
            last_capture_time = current_time

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect checkerboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, CHECKERBOARD,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            # If corners are detected
            if ret:
                print(f"Checkerboard detected for image {captured_images + 1}")
                objpoints.append(objp)

                # Refine corner locations
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners on the frame
                frame_with_corners = cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
                captured_images += 1
                print(f"{captured_images} got ")
                # If 10 valid images are captured, perform calibration
                if captured_images == iteration:
                    print("Performing calibration...")
                    # Perform camera calibration
                    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                    break

                    

        # Display the live feed
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

        # Resize the window to a specific size
        cv2.resizeWindow(name, 600, 400)  # Set desired width and height

        # Show the frame
        cv2.imshow(name, frame)

        # Exit on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cams_Res.append([ret, mtx, dist, rvecs, tvecs])

def afficher_calib(camera_name,mtx, dist):
    st.write(f"**Résultats de la calibration - Caméra {camera_name}:**")
    st.write("\n**Matrice de la caméra :**")
    st.write(mtx)

    st.write("\n**Coefficients de distorsion :**")
    st.write(dist)

    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]

    st.write("\n**Paramètres intrinsèques :**")
    st.write(f"**Distances focales (en pixels) :**")
    st.write(f"- fx = {fx:.2f}")
    st.write(f"- fy = {fy:.2f}")
    st.write(f"**Centre optique (en pixels) :**")
    st.write(f"- cx = {cx:.2f}")
    st.write(f"- cy = {cy:.2f}")

def detect_inrange(image, surface_min, surface_max):
    points = []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)  # Gaussian blur is generally better
    mask = cv2.inRange(blurred, lo, hi)

    # Morphological operations to reduce noise (optional but recommended)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Opening (erosion followed by dilation)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Closing (dilation followed by erosion)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        points.append([int(x), int(y), int(radius), int(area)])
        
    return hsv, mask, points

def object_detection(name, http, num):
    cap = cv2.VideoCapture(http)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la caméra/flux vidéo {http}")
        return  # Return early if camera/stream can't be opened

    # Define the desired output resolution (adjust as needed)
    output_width = 640  # Example: 640 pixels
    output_height = 480 # Example: 480 pixels

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        frame = cv2.resize(frame, (output_width, output_height))

        cv2.flip(frame, 1, frame)
        hsv, mask, points = detect_inrange(frame, 200, 300)  # Assuming detect_inrange is defined elsewhere

        if points:
            x, y, radius, area = points[0]
            cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)
            cv2.putText(frame, f"{x}, {y}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow(f"Image : {name}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def calculer_mi_distance(cam1_extr, cam2_extr):
    """
    Calculate the midpoint between two camera projection centers.
    """
    tvec1 = cam1_extr[4]  # Translation vector for camera 1
    tvec2 = cam2_extr[4]  # Translation vector for camera 2

    mid_point = (tvec1 + tvec2)
    x = mid_point[0][0]/2
    y =mid_point[1][0]/2
    z =mid_point[2][0]/2
    return x , y ,z

st.title("Calibration Caméras")
# Input parameters
with st.form("parameters"):
    distance_cameras = st.number_input("Distance entre les caméras (en mètres)", min_value=0.0, value=1.0, step=0.1, format="%.1f")
    nb_images = st.number_input("Nombre d'images de calibrage", min_value=1, value=10, step=1)
    checkboard_size = st.text_input("Taille du checkboard (ex: '9x6')", value="9x6")
    address_camera1 = st.text_input("Adresse caméra 1 ", value="http://...")
    address_camera2 = st.text_input("Adresse caméra 2 ", value="http://...")
    submitted = st.form_submit_button("Valider")


if submitted : 
    CHECKERBOARD = (int(checkboard_size.split("x")[0]), int(checkboard_size.split("x")[1]))
    iteration = nb_images
    b = distance_cameras
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Camera 1 Feed")
        placeholder1 = st.empty()
    with col2:
        st.subheader("Camera 2 Feed")
        placeholder2 = st.empty()

    """    # cameras calibration multihthreadings
    cam1 = threading.Thread(target=calibrate_camera, args=("cam1" , address_camera1, CHECKERBOARD, iteration))
    cam2 = threading.Thread(target=calibrate_camera, args=("cam2",address_camera2, CHECKERBOARD, iteration))
    cam1.start()
    cam2.start()
    cam1.join()
    cam2.join()
    with cams_res_lock:
        if len(cams_Res) < 2:
            logging.error("Camera calibration failed for one or both cameras.")
            exit(1)
        cam1_results = cams_Res[0]
        cam2_results = cams_Res[1]
    
    col1, col2 = st.columns(2)  # Create two columns

    with col1:
        afficher_calib("camera 1",cams_Res[0][1] , cams_Res[0][2])

    with col2:
        afficher_calib("camera 2",cams_Res[1][1] , cams_Res[1][2])
    
    with col1:
        st.subheader("Camera 1 Feed")
        placeholder1 = st.empty()
    with col2:
        st.subheader("Camera 2 Feed")
        placeholder2 = st.empty()
    # object detection multithreadings"""
    T1 = threading.Thread(target=object_detection, args=("Camera 1", address_camera1, 1))
    T2 = threading.Thread(target=object_detection, args=("Camera 2", address_camera2, 2))
    T1.start()
    T2.start()
    T1.join()
    T2.join()

    """with coords_lock:
        if len(coords) < 2:
            logging.error("Object detection failed for one or both cameras.")
            exit(1)

        if coords[0][1] == 1:
            ul, vl = coords[0][0]
            ur, vr = coords[1][0]
        else:
            ul, vl = coords[1][0]
            ur, vr = coords[0][0]

    fx = cam1_results[1][0, 0]
    fy = cam1_results[1][1, 1]
    ox =cam1_results[1][0, 2]
    oy =cam1_results[1][1, 2]


    x,y,z = calculate_3d_position(ul, vl, ur, fx, fy, b, ox, oy)
    a , b ,c = calculer_mi_distance(cam1_results,cam2_results)
    print(x,y,z)
    print(a,b,c)"""