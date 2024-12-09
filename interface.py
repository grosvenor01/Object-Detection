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
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
cams_Res=[]
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

def boundings_builder(name , image ,  df):
    if image is None or df.empty:
        return image
    for _, row in df.iterrows():
        x1, y1, x2, y2 = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        #calculate center coordinates
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        print(f"X{name} = {center_x}     y{name}= {center_y}")
        cv2.circle(image, (center_x, center_y), 5, (139, 69, 19), -1) # -1 fills the circle

    return image

async def display_camera_feed(camera_source, camera_name, placeholder , CHECKERBOARD ,iteration):
    try:
        # Criteria for corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Vectors to store 3D (real-world) and 2D (image) points
        objpoints = []  # 3D points in real-world space
        imgpoints = []  # 2D points in image plane

        # Real-world 3D coordinates for the checkerboard
        objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            st.error(f"Could not open {camera_name} camera.")
            return
        
        last_capture_time = time.time()
        captured_images = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error(f"{camera_name} camera feed failed.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame)
            placeholder.image(img_pil, channels="RGB", use_container_width=True)
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
            await asyncio.sleep(0.05)  # Use asyncio.sleep
        cams_Res.append([ret, mtx, dist, rvecs, tvecs])
    except Exception as e:
        st.error(f"An error occurred with {camera_name} camera: {e}")

    finally:
        if cap:
            cap.release()

def afficher_calib(mtx,dist):
    print("\nMatrice de la caméra :")
    print(mtx)
    print("\nCoefficients de distorsion :")
    print(dist)


    # Extract intrinsic parameters
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]

                            # Display results
    print("\nFocal lengths (in pixels):")
    print(f"fx = {fx}")
    print(f"fy = {fy}")
    print("\nOptical center (in pixels):")
    print(f"cx = {cx}")
    print(f"cy = {cy}")

st.title("Calibration Caméras")
# Input parameters
with st.form("parameters"):
    distance_cameras = st.number_input("Distance entre les caméras (en mètres)", min_value=0.0, value=1.0, step=0.1, format="%.1f")
    nb_images = st.number_input("Nombre d'images de calibrage", min_value=1, value=10, step=1)
    checkboard_size = st.text_input("Taille du checkboard (ex: '9x6')", value="9x6")
    address_camera1 = st.text_input("Adresse caméra 1 ", value="http://...")
    address_camera2 = st.text_input("Adresse caméra 2 ", value="http://...")
    submitted = st.form_submit_button("Valider")

async def main():
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
        await asyncio.gather(
            display_camera_feed(address_camera1,"cam1",placeholder1,CHECKERBOARD ,iteration),
            display_camera_feed(address_camera2,"cam2",placeholder2,CHECKERBOARD,iteration)
        )

if __name__ == "__main__":
    asyncio.run(main())

