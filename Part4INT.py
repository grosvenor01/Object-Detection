import streamlit as st
import numpy as np
import cv2
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

trajectory = []  # Store trajectory points
trajectory_lock = threading.Lock()

# HSV pour la couleur rose
lo = np.array([140, 50, 50])  # Lower bound (Hue, Saturation, Value)
hi = np.array([170, 255, 255])  # Upper bound (Hue, Saturation, Value)
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import threading
import queue

# Charger le modèle YOLOv8 pré-entraîné
model = YOLO('yolov8s.pt')
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

# Dessiner les zones d’intérêt (ROI)
def draw_roi(frame):
    h, w, _ = frame.shape
    roi_color = (255, 255, 0)
    thickness = 2
    cv2.rectangle(frame, (int(w * 0.2), int(h * 0.2)), (int(w * 0.8), int(h * 0.8)), roi_color, thickness)
    return frame

# Fonction pour capturer les frames vidéo
def capture_frames():
    global frame_count
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
        if area < 200:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        points.append([int(x), int(y), int(radius), int(area)])

    # Track trajectory
    if points:
        with trajectory_lock:
            trajectory.append((points[0][0], points[0][1]))

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

def main():

    # Input form
    with st.form("parameters"):
        address_camera = st.text_input("Adresse de la caméra", value="http://...")
        functionality = st.selectbox("Fonctionnalité", ["Tracking" , "Distance entre 2 objets" , "Détection de types d'objet et Heatmap"])
        submitted = st.form_submit_button("Start")

    if submitted:
        if functionality == "Tracking":
            tracking_thread = threading.Thread(target=object_tracking, args=("Camera", address_camera))
            tracking_thread.start()
        
        elif functionality == "Détection de types d'objet et Heatmap":
            # Démarrer le thread de capture
            capture_thread = threading.Thread(target=capture_frames, daemon=True)
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
            pass


