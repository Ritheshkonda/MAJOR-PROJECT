import time
import csv
import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

from adaptive_engine import AdaptiveEngine

MODEL_PATH = "fine_tuned_vit_tiny"
PROCESSOR_NAME = "facebook/deit-tiny-patch16-224"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CONF_THRESHOLD = 0.35
MAX_DISAPPEARED = 30
LOG_FILE = "emotion_log.csv"
RECOMMENDATIONS_FILE = "recommendations.csv"

ID2LABEL = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

engine = AdaptiveEngine(
    window_seconds=8.0,
    max_records_per_face=200,
    min_confidence=0.25,
)

# ------------------ CSV INITIALIZATION ------------------

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "face_id", "label", "confidence", "x", "y", "w", "h"])

if not os.path.exists(RECOMMENDATIONS_FILE):
    with open(RECOMMENDATIONS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "face_id", "label", "score", "intent", "advice", "priority"])

# ------------------ LOAD MODEL ------------------

print("Device:", DEVICE)
print("Loading model...")

model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
processor = ViTImageProcessor.from_pretrained(PROCESSOR_NAME)

# ------------------ FACE DETECTOR ------------------

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)
if face_detector.empty():
    raise RuntimeError("Haar cascade not found")

# ------------------ CENTROID TRACKER ------------------

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = {}
        self.bboxes = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid, bbox):
        oid = self.next_object_id
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.disappeared[oid] = 0
        self.next_object_id += 1

    def deregister(self, oid):
        self.objects.pop(oid, None)
        self.bboxes.pop(oid, None)
        self.disappeared.pop(oid, None)

    def update(self, rects):
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects, self.bboxes

        input_centroids = [(x + w // 2, y + h // 2) for x, y, w, h in rects]

        if len(self.objects) == 0:
            for i, c in enumerate(input_centroids):
                self.register(c, rects[i])
            return self.objects, self.bboxes

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.linalg.norm(
            np.array(object_centroids)[:, None] - np.array(input_centroids)[None, :],
            axis=2,
        )

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            oid = object_ids[r]
            self.objects[oid] = input_centroids[c]
            self.bboxes[oid] = rects[c]
            self.disappeared[oid] = 0
            used_rows.add(r)
            used_cols.add(c)

        for r in set(range(len(object_ids))) - used_rows:
            oid = object_ids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        for c in set(range(len(input_centroids))) - used_cols:
            self.register(input_centroids[c], rects[c])

        return self.objects, self.bboxes

tracker = CentroidTracker(MAX_DISAPPEARED)

# ------------------ CAMERA ------------------

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

prev_time = time.time()
fps = 0.0

# ------------------ MAIN LOOP ------------------

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.05, 6, minSize=(60, 60))
        rects = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

        _, bboxes = tracker.update(rects)

        for fid, (x, y, w, h) in bboxes.items():
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue

            image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=-1)
                conf, pred = probs.max(dim=-1)

            label = ID2LABEL[int(pred)]
            conf_val = float(conf.item())

            engine.update(fid, label, conf_val, time.time())
            rec = engine.get_recommendation(fid)

            advice = rec["advice"].replace("—", "-").replace("–", "-")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"ID:{fid} {label}", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            if rec["priority"] <= 2 and rec["score"] >= 0.4:
                cv2.putText(frame, advice[:60], (10, FRAME_HEIGHT-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

            ts = datetime.utcnow().isoformat()

            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([ts, fid, label, f"{conf_val:.4f}", x, y, w, h])

            with open(RECOMMENDATIONS_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    ts, fid, rec["label"], rec["score"],
                    rec["intent"], advice, rec["priority"]
                ])

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-8))
        prev_time = curr_time

        cv2.putText(frame, f"FPS:{fps:.1f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Adaptive Multi-Face Emotion (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")
