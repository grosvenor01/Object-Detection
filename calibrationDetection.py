import torch
import cv2
import numpy as np
import pandas as pd
import pathlib
from IPython.display import display
import time
import threading

#bonus code (usedfor the relatives path finding)
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

# build the boundings of a specific class
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
    cams_Res.append([ret, mtx, dist, rvecs, tvecs])

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

def detect_object(name , http):
    capture = cv2.VideoCapture(http)
    if not capture.isOpened():
        print("Error opening video stream.")
        exit()
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 600, 400)
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error reading frame.")
            break
        results = model(frame)
        df = stacking_results(results)
        image = boundings_builder(name , frame, df)
        cv2.imshow(name, image)
        if cv2.waitKey(1) == ord("q"):
            break
    time.sleep(10)


# initialization
http1 = 'http://192.168.137.244:8080/video'
http2 = 'https://192.168.137.244:8080/video'
CHECKERBOARD = (9, 7)
iteration = 10 

# load object detection model detection 
model = load_models()

# cameras calibration multihthreadings
cam1 = threading.Thread(target=calibrate_camera, args=("cam1" , http1, CHECKERBOARD, 10))
cam2 = threading.Thread(target=calibrate_camera, args=("cam2",http2, CHECKERBOARD, 10))
cam1.start()
cam2.start()
cam1.join()
cam2.join()

# Display calibration results
cam1 = cams_Res[0]
cam2 = cams_Res[1]
afficher_calib(cam1[1] , cam1[2])
afficher_calib(cam2[1] , cam2[2])

# object detection multithreadings
T1 = threading.Thread(target=detect_object, args=("camera 1" , http1,))
T2 = threading.Thread(target=detect_object, args=("camera 2" , http2,))
T1.start()
T2.start()
T1.join()
T2.join()