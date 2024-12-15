import streamlit as st
import numpy as np
import cv2
import threading
import logging
import pathlib
import math
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

trajectory = []  # Store trajectory points
trajectory_lock = threading.Lock()

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# HSV range for pink
lo = np.array([140, 50, 50])  # Lower bound (Hue, Saturation, Value)
hi = np.array([170, 255, 255])  # Upper bound (Hue, Saturation, Value)

cams_Res=[]
coords = []

pos_cam1 = None 
pos_cam2 = None 
lock = threading.Lock()
cams_res_lock = threading.Lock()
coords_lock = threading.Lock()


from ultralytics import YOLO
from collections import defaultdict
import time
import threading
import queue

# Charger le modèle YOLOv8 pré-entraîné
model = YOLO('yolov8s.pt')  # 'yolov8s.pt' est un modèle léger pour la détection rapide
# Capture vidéo
videoCap = cv2.VideoCapture(0)

# Initialisation des variables pour le suivi et les statistiques
object_tracker = defaultdict(list)
frame_count = 0
start_time = time.time()
stats = defaultdict(int)

# Variables de gestion du multithreading
frame_queue = queue.Queue(maxsize=1)

# Variables pour la heatmap
heatmap_accumulator = None


def calculate_3d_position(uL, vL, uR, fx, fy, b, ox, oy):
    uL = uL + 0.05
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


# Dessiner les zones d’intérêt (ROI)
def draw_roi(frame):
    h, w, _ = frame.shape
    roi_color = (255, 255, 0)
    thickness = 2
    cv2.rectangle(frame, (int(w * 0.2), int(h * 0.2)), (int(w * 0.8), int(h * 0.8)), roi_color, thickness)
    return frame

# Fonction pour capturer les frames vidéo
def capture_frames(adress):
    global frame_count
    videoCap = cv2.VideoCapture(adress)
    while True:
        ret, frame = videoCap.read()
        if not ret:
            break
        frame_count += 1
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)

# Fonction pour mettre à jour la heatmap
def update_heatmap(heatmap, bbox, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    heatmap[y1:y2, x1:x2] += 1
    return heatmap

def detect_and_track(image):
    """Detects the object and tracks its trajectory."""
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
        if area < 200:  # Ignore small noise contours
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        points.append([int(x), int(y), int(radius), int(area)])

    # Track trajectory
    if points:
        with trajectory_lock:
            trajectory.append((points[0][0], points[0][1]))  # Append only the largest object's center

    return mask, points

def draw_trajectory(frame):
    """Draws the trajectory as a line on the frame."""
    with trajectory_lock:
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)  # Green line

def object_tracking(name, http):
    """Tracks the object in the video stream."""
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
        mask, points = detect_and_track(frame)

        # Draw detected points and trajectory
        if points:
            x, y, radius, area = points[0]
            cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)  # Red circle for detected object
            cv2.putText(frame, f"{x}, {y}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        draw_trajectory(frame)  # Draw trajectory line

        cv2.imshow(f"{name}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

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

    # Process the two largest objects (if at least two exist)
    for i, contour in enumerate(contours[:2]):  # Take the first two contours
        area = cv2.contourArea(contour)
        
        (x, y), radius = cv2.minEnclosingCircle(contour)
        points.append([int(x), int(y), int(radius), int(area)])  # Append only the x and y coordinates

    
    return hsv, mask, points  # Fill missing points with None if fewer than two objects are detected

def calculate_3d_distance(point1, point2):

    x1, y1, z1 = point1
    x2, y2, z2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

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

def object_detection(name, http, num, cam1_results, cam2_results, distance_cameras):
    global pos_cam1, pos2_cam1, pos_cam2, pos2_cam2
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

        with lock:  # Protect shared variables with a lock
            if len(points) >= 2:  # Ensure at least two objects are detected
                x, y, radius, area = points[0]
                x2, y2, radius2, area2 = points[1]
                cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)
                cv2.circle(frame, (x2, y2), radius2, (0, 0, 255), 2)
                cv2.putText(frame, f"{x}, {y}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f"{x2}, {y2}", (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Store positions for the respective camera
                if num == 1:
                    pos_cam1 = (x, y)
                    pos2_cam1 = (x2, y2)
                else:
                    pos_cam2 = (x, y)
                    pos2_cam2 = (x2, y2)

                # Calculate 3D position if all points are available
                if (
                    pos_cam1 is not None and pos2_cam1 is not None and
                    pos_cam2 is not None and pos2_cam2 is not None
                ):
                    # First object
                    ul, vl = pos_cam1
                    ur, vr = pos_cam2
                    fx = cam1_results[1][0, 0]
                    fy = cam1_results[1][1, 1]
                    ox = cam1_results[1][0, 2]
                    oy = cam1_results[1][1, 2]
                    b = distance_cameras

                    x1, y1, z = calculate_3d_position(ul, vl, ur, fx, fy, b, ox, oy)
                    cv2.putText(frame, f"{x1:.2f}, {y1:.2f}, {z:.2f}",
                                (x + 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Second object
                    ul2, vl2 = pos2_cam1
                    ur2, vr2 = pos2_cam2
                    x3, y3, z2 = calculate_3d_position(ul2, vl2, ur2, fx, fy, b, ox, oy)
                    cv2.putText(frame, f"{x3:.2f}, {y3:.2f}, {z2:.2f}",
                                (x2 + 20, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Calculate and display distance
                    distance = calculate_3d_distance((x1, y1, z), (x3, y3, z2))
                    distance_text = f"Distance: {distance:.2f} units"
                    cv2.putText(frame, distance_text, (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(f"Image: {name}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():

    # Input form
    with st.form("parameters"):

        address_camera = st.text_input("Adresse de la caméra", value="http://...")
        functionality = st.selectbox("Fonctionnalités", ["Tracking" , "Distance entre 2 objets" , "Détection de types d'objets et Heatmap"])
        
        st.write("Paramètres pour la distance entre 2 objets")
        distance_cameras = st.number_input("Distance entre les caméras (en mètres)", min_value=0.0, value=1.0, step=0.1, format="%.1f")
        nb_images = st.number_input("Nombre d'images de calibrage", min_value=1, value=10, step=1)
        checkboard_size = st.text_input("Taille du checkboard (ex: '9x7')", value="9x7")
        address_camera2 = st.text_input("Adresse caméra 2 ", value="http://..")
        model_case = st.selectbox("Type de calibration" , ["fixe object" , "fixe camera"])
        
        submitted = st.form_submit_button("Start")

    if submitted:
        if functionality == "Tracking":
            tracking_thread = threading.Thread(target=object_tracking, args=("Camera", address_camera))
            tracking_thread.start()
        
        elif functionality == "Détection de types d'objets et Heatmap":
            # Démarrer le thread de capture
            capture_thread = threading.Thread(target=capture_frames, args = (address_camera,), daemon=True)
            capture_thread.start()
            global heatmap_accumulator
            while True:
                if not frame_queue.empty():
                    frame = frame_queue.get()
                    h, w, _ = frame.shape

                    # Initialiser la heatmap si elle n'existe pas encore
                    if heatmap_accumulator is None:
                        heatmap_accumulator = np.zeros((h, w), dtype=np.float32)

                    # Détection des objets
                    results = model(frame, device='cpu')

                    # Variables locales
                    roi_object_count = 0

                    for result in results:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        categories = result.boxes.cls.cpu().numpy().astype(int)
                        labels = [model.names[category] for category in categories]

                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = map(int, box)
                            confidence = confidences[i]
                            label = labels[i]

                            # Vérifier si l’objet est dans la ROI
                            if x1 > int(w * 0.2) and x2 < int(w * 0.8) and y1 > int(h * 0.2) and y2 < int(h * 0.8):
                                roi_object_count += 1
                                object_tracker[label].append((x1, y1, x2, y2))
                                stats[label] += 1

                                # Dessiner la boîte englobante
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{label} ({confidence:.2f})", 
                                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                                # Mise à jour de la heatmap
                                heatmap_accumulator = update_heatmap(heatmap_accumulator, (x1, y1, x2, y2), frame.shape)

                    # Calculer et afficher le FPS
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Afficher les zones d’intérêt
                    frame = draw_roi(frame)

                    # Normaliser la heatmap et appliquer une coloration
                    heatmap_normalized = cv2.normalize(heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

                    # Redimensionner pour l’affichage
                    resized_frame = cv2.resize(frame, (640, 480))
                    resized_heatmap = cv2.resize(heatmap_color, (640, 480))

                    # Afficher les deux fenêtres
                    cv2.imshow("Detection d'objets avec YOLOv8", resized_frame)
                    cv2.imshow("Heatmap des detections", resized_heatmap)

                    # Quitter avec 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            # Affichage des statistiques finales
            print("Statistiques de détection :")
            for label, count in stats.items():
                print(f"{label}: {count}")

            # Libération des ressources
            videoCap.release()
            cv2.destroyAllWindows()
        
        elif functionality == "Distance entre 2 objets": 
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
            cam1 = threading.Thread(target=calibrate_camera, args=("cam1" , address_camera, CHECKERBOARD, iteration))
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


            

            if model_case == "fixe camera":
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
                threading.Thread(target=monitor_and_recalibrate, args=("cam1", address_camera, cam1_results, 0), daemon=True).start()
                threading.Thread(target=monitor_and_recalibrate, args=("cam2", address_camera2, cam2_results, 1), daemon=True).start()
                
            # object detection multithreadings"""
            T1 = threading.Thread(target=object_detection, args=("Camera 1", address_camera, 1 , cam1_results,cam2_results, distance_cameras))
            T2 = threading.Thread(target=object_detection, args=("Camera 2", address_camera2, 2 , cam1_results,cam2_results, distance_cameras))
            T1.start()
            T2.start()
            T1.join()
            T2.join()



