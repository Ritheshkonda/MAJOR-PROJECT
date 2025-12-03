# realtime_face_logging.py
import time
import csv
import os
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTImageProcessor
from datetime import datetime

# CONFIG
MODEL_PATH = "fine_tuned_vit_tiny"
PROCESSOR_NAME = "facebook/deit-tiny-patch16-224"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONF_THRESHOLD = 0.35
SMOOTH_WINDOW = 5
LOG_FILE = "emotion_log.csv"

ID2LABEL = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# prepare log file
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "face_id", "label", "confidence", "x", "y", "w", "h"])

print("Device:", DEVICE)
print("Loading model...")
model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
processor = ViTImageProcessor.from_pretrained(PROCESSOR_NAME)

# face detector (Haar cascade provided with OpenCV)
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(face_cascade_path)

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# small tracking: maintain previous centroids -> assign IDs
next_face_id = 0
prev_faces = {}  # id -> centroid
face_smooth = {}  # id -> deque of last labels/confidences

def assign_ids(detected_boxes, prev_faces, max_distance=80):
    global next_face_id
    assigned = {}
    used_prev = set()
    for i, (x, y, w, h) in enumerate(detected_boxes):
        cx, cy = x + w//2, y + h//2
        best_id = None
        best_dist = None
        for fid, (pcx, pcy) in prev_faces.items():
            d = (pcx - cx)**2 + (pcy - cy)**2
            if best_dist is None or d < best_dist:
                best_dist = d
                best_id = fid
        if best_id is not None and best_dist <= max_distance**2 and best_id not in used_prev:
            assigned[i] = best_id
            used_prev.add(best_id)
        else:
            assigned[i] = next_face_id
            next_face_id += 1
    return assigned

prev_time = time.time()
fps = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        detected = list(faces)
        assigned = assign_ids(detected, prev_faces)

        new_prev = {}
        for idx, (x, y, w, h) in enumerate(detected):
            face_id = assigned[idx]
            cx, cy = x + w//2, y + h//2
            new_prev[face_id] = (cx, cy)

            # crop, preprocess, predict
            face_img = rgb[y:y+h, x:x+w]
            if face_img.size == 0:
                continue
            # resize to processor expectation via PIL-like handling by processor
            from PIL import Image
            pil_face = Image.fromarray(face_img).convert("RGB")
            inputs = processor(images=pil_face, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                conf, pred = torch.max(probs, dim=-1)
                pred_idx = int(pred.item())
                conf_val = float(conf.item())

            # smoothing per face id
            dq = face_smooth.get(face_id, deque(maxlen=SMOOTH_WINDOW))
            dq.append((pred_idx, conf_val))
            face_smooth[face_id] = dq
            # majority label and avg conf
            labels_arr = np.array([p for p,c in dq])
            conf_arr = np.array([c for p,c in dq])
            vals, counts = np.unique(labels_arr, return_counts=True)
            maj_idx = int(vals[np.argmax(counts)])
            avg_conf = float(conf_arr.mean()) if len(conf_arr)>0 else conf_val
            label = ID2LABEL.get(maj_idx, str(maj_idx))
            display = f"{label} {avg_conf:.2f}" if avg_conf >= CONF_THRESHOLD else f"Uncertain {avg_conf:.2f}"

            # draw
            color = (0,255,0) if avg_conf>=CONF_THRESHOLD else (0,165,255)
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"ID:{face_id} {display}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # log each detection
            ts = datetime.utcnow().isoformat()
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ts, face_id, label, f"{avg_conf:.4f}", x, y, w, h])

        prev_faces = new_prev

        # FPS
        curr_time = time.time()
        fps = 0.9*fps + 0.1*(1.0/(curr_time - prev_time + 1e-8))
        prev_time = curr_time
        cv2.putText(frame, f"FPS:{fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

        cv2.imshow("Real-time Face Emotion Logging (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped")
