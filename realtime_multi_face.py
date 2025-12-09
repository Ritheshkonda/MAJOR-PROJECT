import time
import csv
import os
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

# ---------- CONFIG ----------
MODEL_PATH = "fine_tuned_vit_tiny"
PROCESSOR_NAME = "facebook/deit-tiny-patch16-224"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CONF_THRESHOLD = 0.35
SMOOTH_WINDOW = 5            # frames per-face to smooth predictions
MAX_DISAPPEARED = 30         # frames to wait before deregistering a face
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

# ---------- UTIL: CentroidTracker (register/deregister + nearest matching) ----------
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = dict()          # object_id -> centroid (x,y)
        self.bboxes = dict()           # object_id -> bbox (x,y,w,h)
        self.disappeared = dict()      # object_id -> consecutive disappeared frames
        self.max_disappeared = max_disappeared

    def register(self, centroid, bbox):
        oid = self.next_object_id
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.disappeared[oid] = 0
        self.next_object_id += 1
        return oid

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.bboxes:
            del self.bboxes[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, rects):
        # rects: list of (x,y,w,h)
        if len(rects) == 0:
            # mark all as disappeared
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects, self.bboxes

        input_centroids = []
        for (x, y, w, h) in rects:
            cx = int(x + w/2)
            cy = int(y + h/2)
            input_centroids.append((cx, cy))

        # if no existing objects, register all
        if len(self.objects) == 0:
            for i, c in enumerate(input_centroids):
                self.register(c, rects[i])
            return self.objects, self.bboxes

        # build arrays of existing centroids
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # compute distance matrix (existing x new)
        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=np.float32)
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = (oc[0]-ic[0])**2 + (oc[1]-ic[1])**2

        # greedy matching: find smallest distances
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for (r, c) in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            oid = object_ids[r]
            self.objects[oid] = input_centroids[c]
            self.bboxes[oid] = rects[c]
            self.disappeared[oid] = 0
            used_rows.add(r)
            used_cols.add(c)

        # check for unmatched existing object rows -> mark disappeared
        unmatched_rows = set(range(0, D.shape[0])) - used_rows
        for r in unmatched_rows:
            oid = object_ids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        # register unmatched new input cols
        unmatched_cols = set(range(0, D.shape[1])) - used_cols
        for c in unmatched_cols:
            self.register(input_centroids[c], rects[c])

        return self.objects, self.bboxes

# ---------- Prepare logging ----------
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "face_id", "label", "confidence", "x", "y", "w", "h"])

# ---------- Load model & processor ----------
print("Device:", DEVICE)
print("Loading model...")
model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
processor = ViTImageProcessor.from_pretrained(PROCESSOR_NAME)

# per-face smoothing store: face_id -> deque of (label_idx, confidence)
face_smooth = dict()

# prepare face detector (Haar)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)
if face_detector.empty():
    raise RuntimeError("Haar cascade not found in OpenCV installation.")

# video capture
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

tracker = CentroidTracker(max_disappeared=MAX_DISAPPEARED)
prev_time = time.time()
fps = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))
        rects = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

        objects, bboxes = tracker.update(rects)

        # iterate tracked faces
        for fid, bbox in bboxes.items():
            x, y, w, h = bbox
            # clamp bbox within frame
            x, y = max(0, x), max(0, y)
            w, h = max(10, w), max(10, h)
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue
            pil_face = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).convert("RGB")

            # predict
            inputs = processor(images=pil_face, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                conf_val, pred = torch.max(probs, dim=-1)
                pred_idx = int(pred.item())
                conf_val = float(conf_val.item())

            # smoothing
            dq = face_smooth.get(fid, deque(maxlen=SMOOTH_WINDOW))
            dq.append((pred_idx, conf_val))
            face_smooth[fid] = dq

            labels_arr = np.array([p for p,c in dq]) if len(dq)>0 else np.array([pred_idx])
            conf_arr = np.array([c for p,c in dq]) if len(dq)>0 else np.array([conf_val])
            vals, counts = np.unique(labels_arr, return_counts=True)
            maj_idx = int(vals[np.argmax(counts)])
            avg_conf = float(conf_arr.mean())

            label = ID2LABEL.get(maj_idx, str(maj_idx))
            display_label = f"{label} {avg_conf:.2f}" if avg_conf >= CONF_THRESHOLD else f"Uncertain {avg_conf:.2f}"

            color = (0,255,0) if avg_conf >= CONF_THRESHOLD else (0,165,255)
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"ID:{fid} {display_label}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # log
            ts = datetime.utcnow().isoformat()
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ts, fid, label, f"{avg_conf:.4f}", x, y, w, h])

        # clean smoothing entries for deregistered ids
        tracked_ids = set(bboxes.keys())
        for fid in list(face_smooth.keys()):
            if fid not in tracked_ids:
                # keep smoothing state for a while; if object disappeared permanently, remove
                # if tracker no longer has it, deregister will have happened; remove smoothing
                if fid not in tracker.objects:
                    del face_smooth[fid]

        # compute fps
        curr_time = time.time()
        fps = 0.9*fps + 0.1*(1.0/(curr_time - prev_time + 1e-8))
        prev_time = curr_time
        cv2.putText(frame, f"FPS:{fps:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

        cv2.imshow("Multi-Face Real-time Emotion (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")
