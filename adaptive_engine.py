# adaptive_engine.py
import time
from collections import deque, defaultdict, Counter
from typing import Dict, List, Tuple, Any

DEFAULT_INTENTS = {
    "angry":    {"state": "frustrated", "priority": 3, "advice": "Slow down; offer a break and rephrase."},
    "disgust":  {"state": "uncomfortable", "priority": 3, "advice": "Change examples; check clarity."},
    "fear":     {"state": "confused", "priority": 3, "advice": "Provide reassurance and simpler examples."},
    "happy":    {"state": "engaged", "priority": 1, "advice": "Students engaged — continue or increase difficulty."},
    "neutral":  {"state": "neutral", "priority": 2, "advice": "Consider a quick interactive question."},
    "sad":      {"state": "low_mood", "priority": 3, "advice": "Offer encouragement; reduce cognitive load."},
    "surprise": {"state": "attentive", "priority": 1, "advice": "Good engagement — follow up with quick question."},
}

class AdaptiveEngine:
    """
    Maintains per-face rolling history and produces recommendations.
    Usage:
      engine = AdaptiveEngine(window_seconds=8, fps=5)
      engine.update(face_id, label, confidence, timestamp=time.time())
      rec = engine.get_recommendation(face_id)
      class_recs = engine.summary()  # optional aggregated view
    """

    def __init__(self,
                 window_seconds: float = 8.0,
                 max_records_per_face: int = 100,
                 intents: Dict[str, Dict] = None,
                 min_confidence: float = 0.25):
        self.window_seconds = window_seconds
        self.max_records_per_face = max_records_per_face
        self.intents = intents or DEFAULT_INTENTS
        self.min_confidence = min_confidence

        # per-face deque of (timestamp, label, confidence)
        self._hist = defaultdict(deque)

    def update(self, face_id: int, label: str, confidence: float, timestamp: float = None):
        """
        Append a detection for face_id. Automatically prunes old entries outside window_seconds.
        """
        ts = timestamp or time.time()
        dq = self._hist[face_id]
        dq.append((ts, label, float(confidence)))
        while len(dq) > self.max_records_per_face:
            dq.popleft()
        cutoff = ts - self.window_seconds
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _get_recent(self, face_id: int) -> List[Tuple[float, str, float]]:
        return list(self._hist.get(face_id, []))

    def dominant_emotion(self, face_id: int) -> Tuple[str, float]:
        
        recent = [r for r in self._get_recent(face_id) if r[2] >= self.min_confidence]
        if not recent:
            return ("unknown", 0.0)
        labels = [r[1] for r in recent]
        counts = Counter(labels)
        label, cnt = counts.most_common(1)[0]
        score = cnt / len(recent)
        return (label, score)

    def get_recommendation(self, face_id: int) -> Dict[str, Any]:
        """
        Returns a structured recommendation for a face_id:
        {
          "face_id": int,
          "label": str,
          "score": float,
          "intent": "confused",
          "advice": "...",
          "priority": int,
          "window_count": int
        }
        """
        label, score = self.dominant_emotion(face_id)
        recent = self._get_recent(face_id)
        window_count = len(recent)
        if label in self.intents:
            intent = self.intents[label]
            rec = {
                "face_id": face_id,
                "label": label,
                "score": round(score, 3),
                "intent": intent.get("state"),
                "advice": intent.get("advice"),
                "priority": intent.get("priority"),
                "window_count": window_count,
                "timestamp": time.time()
            }
        else:
            rec = {
                "face_id": face_id,
                "label": label,
                "score": round(score, 3),
                "intent": "unknown",
                "advice": "",
                "priority": 99,
                "window_count": window_count,
                "timestamp": time.time()
            }
        return rec

    def summary(self) -> Dict[str, Any]:
        """
        Aggregated classroom summary: counts of dominant labels and high-priority flags.
        """
        doms = []
        for fid in list(self._hist.keys()):
            lab, score = self.dominant_emotion(fid)
            doms.append(lab)
        counts = Counter(doms)
        negative = sum(counts.get(k, 0) for k in ["fear", "sad", "angry", "disgust"])
        positive = sum(counts.get(k, 0) for k in ["happy", "surprise"])
        total_tracked = len(doms)
        engagement = None
        if total_tracked > 0:
            engagement = (positive - negative) / total_tracked  # can be -1..1
        return {
            "dominant_counts": counts,
            "negative_count": negative,
            "positive_count": positive,
            "tracked": total_tracked,
            "engagement_score": engagement
        }

    def clear_face(self, face_id: int):
        if face_id in self._hist:
            del self._hist[face_id]

    def clear_all(self):
        self._hist.clear()
