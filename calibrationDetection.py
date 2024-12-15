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
# HSV range for pink (adjust these values if needed)
lo = np.array([140, 50, 50])  # Lower bound (Hue, Saturation, Value)
hi = np.array([170, 255, 255])  # Upper bound (Hue, Saturation, Value)

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

def object_detction(name , http):
    cap = cv2.VideoCapture(http)
    while (True):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.flip(frame, 1, frame)
        hsv, mask, points = detect_inrange(frame, 200, 300)

        if points:
            x, y, radius, area = points[0]
            cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)  # Draw circle
            cv2.putText(frame, f"{x} , {y}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Mask", mask)
        cv2.imshow("Image", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


# initialization
http1 = 'http://192.168.137.244:8080/video'
http2 = 'https://192.168.137.244:8080/video'
CHECKERBOARD = (9, 7)
iteration = 10 

"""# cameras calibration multihthreadings
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
afficher_calib(cam2[1] , cam2[2])"""

# object detection multithreadings
T1 = threading.Thread(target=object_detction, args=("camera 1" , http1,))
T2 = threading.Thread(target=object_detction, args=("camera 2" , http2,))
T1.start()
T2.start()
T1.join()
T2.join()