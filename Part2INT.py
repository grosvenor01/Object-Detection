import streamlit as st
import numpy as np
import cv2
import numpy as np
import pathlib
import time
import threading
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
cams_Res=[]
coords = []
cams_res_lock = threading.Lock()
coords_lock = threading.Lock()


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
        if current_time - last_capture_time >= 3:
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

    return ret, mtx, dist, rvecs, tvecs

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

st.title("Calibration Caméras")
# Input parameters
with st.form("parameters"):
    distance_cameras = st.number_input("Distance entre les caméras (en mètres)", min_value=0.0, value=1.0, step=0.1, format="%.1f")
    nb_images = st.number_input("Nombre d'images de calibrage", min_value=1, value=10, step=1)
    checkboard_size = st.text_input("Taille du checkboard (ex: '9x6')", value="9x6")
    address_camera1 = st.text_input("Adresse caméra 1 ", value="http://...")
    submitted = st.form_submit_button("Valider")


if submitted : 
    CHECKERBOARD = (int(checkboard_size.split("x")[0]), int(checkboard_size.split("x")[1]))
    iteration = nb_images
    b = distance_cameras

    calib_result = calibrate_camera("cam1", address_camera1, CHECKERBOARD, iteration)
    afficher_calib("cam 1" , calib_result[1] , calib_result[2])