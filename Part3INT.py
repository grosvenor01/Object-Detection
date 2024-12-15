import streamlit as st
import numpy as np
import cv2
import numpy as np
import pathlib
import time
import threading
import streamlit as st
import logging
import torch
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
cams_Res=[]
coords = []

pos_cam1 = None 
pos_cam2 = None 
lock = threading.Lock()
cams_res_lock = threading.Lock()
coords_lock = threading.Lock()


# HSV range for pink (adjust these values if needed)
lo = np.array([140, 50, 50])  # Lower bound (Hue, Saturation, Value)
hi = np.array([170, 255, 255])  # Upper bound (Hue, Saturation, Value)

def calculate_3d_position(uL, vL, uR, fx, fy, b, ox, oy):
    if uL == uR:
        uL = uL +0.05
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

def object_detection_color(name, http, num, cam1_results, cam2_results, distance_cameras):
    global pos_cam1
    global pos_cam2
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

        with lock: # Protect shared variables with a lock
            if points:
                x, y, radius, area = points[0]
                cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)
                cv2.putText(frame, f"{x}, {y}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                if num == 1:
                    pos_cam1 = (x, y)
                else:
                    pos_cam2 = (x, y)

                # Calculate 3D position ONLY if both cameras have detected objects
                if pos_cam1 is not None  and pos_cam2 is not None:
                    ul, vl = pos_cam1
                    ur, vr = pos_cam2
                    fx = cam1_results[1][0, 0]
                    fy = cam1_results[1][1, 1]
                    ox = cam1_results[1][0, 2]
                    oy = cam1_results[1][1, 2]
                    b = distance_cameras 

                    x1, y1, z = calculate_3d_position(ul, vl, ur, fx, fy, b, ox, oy)
                    cv2.putText(frame, f"{x:.2f}, {y:.2f}, {z:.2f}", (x+20 , y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow(f"Image : {name}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def calculer_mi_distance(cam1_extr, cam2_extr):
    tvec1 = cam1_extr[4]  # Translation vector for camera 1
    tvec2 = cam2_extr[4]  # Translation vector for camera 2

    mid_point = (tvec1 + tvec2)
    x = mid_point[0][0]/2
    y =mid_point[1][0]/2
    z =mid_point[2][0]/2
    return x , y ,z

def check_camera_moved(reference_frame, current_frame, threshold=20):
    # Conversion en niveaux de gris
    ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Détection des points caractéristiques (ORB)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    
    # Correspondance des points caractéristiques
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Calcul de la distance moyenne des points correspondants
    distances = [match.distance for match in matches]
    avg_distance = np.mean(distances) if distances else float('inf')
    print(f"{avg_distance} , {threshold}")
    # Si la distance moyenne dépasse le seuil, la caméra a bougé
    return avg_distance > threshold


# Load model once at the start
def load_models():
    try:
        # Load model
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'last.pt', force_reload=True, trust_repo=True)
        device = torch.device('cpu')
        model.to(device)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to convert model results to DataFrame
def stacking_results(model_results):
    return model_results.pandas().xyxy[0] if model_results else pd.DataFrame()

def boundings_builder(image, df):
    if image is None or df.empty:
        return image

    # Separate detections into class 0 and class 1
    class_0 = df[df["class"] == 0]
    class_1 = df[df["class"] == 1]

    if class_0.empty or class_1.empty:
        # If either class is not detected, just draw bounding boxes
        for _, row in df.iterrows():
            x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image

    # Get center coordinates of class 0 and class 1 objects
    centers_0 = [(int((row["xmin"])), int((row["ymax"]))) for _, row in class_0.iterrows()]
    centers_1 = [(int((row["xmin"] + row["xmax"]) / 2), int((row["ymin"] + row["ymax"]) / 2)) for _, row in class_1.iterrows()]

    # Draw bounding boxes for all objects
    for _, row in df.iterrows():
        x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Calculate relative coordinates for each class 1 object with respect to the nearest class 0
    for (x1, y1) in centers_1:
        # Find the nearest class 0 object
        nearest_0 = min(centers_0, key=lambda c0: ((x1 - c0[0]) * 2 + (y1 - c0[1]) * 2))

        # Calculate relative coordinates
        relative_x = x1 - nearest_0[0]
        relative_y = nearest_0[1] - y1 

        # Display relative coordinates on the image
        cv2.putText(image, f"Rel: ({relative_x}, {relative_y})", (x1 + 10, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image


def Yolo_detection(address):
    model = load_models()
    capture = cv2.VideoCapture(address)
    if not capture.isOpened():
        print("Error opening video stream.")
        exit()
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
        cv2.imshow(f"livestreem", frame_resized)
        # Exit on pressing 'q'
        if cv2.waitKey(1) == ord("q"):
            break

def main():
    st.title("Calibration Caméras")
    # Input parameters
    with st.form("parameters"):
        distance_cameras = st.number_input("Distance entre les caméras (en mètres)", min_value=0.0, value=1.0, step=0.1, format="%.1f")
        nb_images = st.number_input("Nombre d'images de calibrage", min_value=1, value=10, step=1)
        checkboard_size = st.text_input("Taille du checkboard (ex: '9x7')", value="9x7")
        address_camera1 = st.text_input("Adresse caméra 1 ", value="http://...")
        address_camera2 = st.text_input("Adresse caméra 2 ", value="http://...")
        model_case = st.selectbox("calibration type" , ["fixe object" , "fixe camera"])
        model_used = st.selectbox("calibration type" , ["Color Detection" , "YOLO"])
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

        # cameras calibration multihthreadings
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
        
        a , b ,c = calculer_mi_distance(cam1_results,cam2_results)
        st.write(f"la position de camera a mi distance est :({a} , {b} , {c})" )

        if model_case == "fixe object":
            def monitor_and_recalibrate(camera_name, address, cam_results, cam_index):
                reference_frame = None
                
                # Capture d'une image de référence lors du premier calibrage
                cap = cv2.VideoCapture(address)
                if cap.isOpened():
                    ret, reference_frame = cap.read()
                    cap.release()
                
                while True:
                    cap = cv2.VideoCapture(address)
                    if not cap.isOpened():
                        break
                    
                    ret, current_frame = cap.read()
                    cap.release()
                    
                    if ret and reference_frame is not None:
                        if check_camera_moved(reference_frame, current_frame):
                            logging.info(f"{camera_name} moved. Recalibrating...")
                            calibrate_camera(camera_name, address, CHECKERBOARD, iteration)
                            with cams_res_lock:
                                cams_Res[cam_index] = cams_Res[-1]
                                cams_Res.pop()
                            # Mettre à jour la nouvelle image de référence
                            reference_frame = current_frame
                            time.sleep(100)

            # Lancer les threads de surveillance pour chaque caméra
            threading.Thread(target=monitor_and_recalibrate, args=("cam1", address_camera1, cam1_results, 0), daemon=True).start()
            threading.Thread(target=monitor_and_recalibrate, args=("cam2", address_camera2, cam2_results, 1), daemon=True).start()
            
        # object detection multithreadings"""
        if model_used == "YOLO":
            T1 = threading.Thread(target=Yolo_detection, args=(address_camera1,) , daemon=True)
            T1.start()
            T1.join()
        else :
            T1 = threading.Thread(target=object_detection_color, args=("Camera 1", address_camera1, 1 , cam1_results,cam2_results, distance_cameras))
            T2 = threading.Thread(target=object_detection_color, args=("Camera 2", address_camera2, 2 , cam1_results,cam2_results, distance_cameras))
            T1.start()
            T2.start()
            T1.join()
            T2.join()

if __name__ == "__main__":
    main()
