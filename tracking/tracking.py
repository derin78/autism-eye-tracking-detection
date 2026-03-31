"""
Improved eye-tracking module with iris-based gaze, EMA smoothing,
blink detection, and head-pose compensation.
"""

import csv
import os
import sys
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

# Allow imports when running as script or as package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ═══════════════════════════════════════════════════════════════════
# GazeData – lightweight container for one frame's results
# ═══════════════════════════════════════════════════════════════════
class GazeData:
    """Container for a single frame of gaze information."""

    __slots__ = (
        "timestamp",
        "left_iris", "right_iris", "iris_center",
        "gaze_screen_x", "gaze_screen_y",
        "left_ear", "right_ear", "blink",
        "head_yaw", "head_pitch",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def csv_row(self):
        return [
            self.timestamp,
            *(self.left_iris  if self.left_iris  is not None else (None, None)),
            *(self.right_iris if self.right_iris is not None else (None, None)),
            *(self.iris_center if self.iris_center is not None else (None, None)),
            self.gaze_screen_x, self.gaze_screen_y,
            self.left_ear, self.right_ear,
            int(self.blink) if self.blink is not None else None,
            self.head_yaw, self.head_pitch,
        ]

    @staticmethod
    def csv_header():
        return [
            "timestamp",
            "left_iris_x", "left_iris_y",
            "right_iris_x", "right_iris_y",
            "iris_center_x", "iris_center_y",
            "gaze_screen_x", "gaze_screen_y",
            "left_ear", "right_ear",
            "blink",
            "head_yaw", "head_pitch",
        ]


# ═══════════════════════════════════════════════════════════════════
# GazeTracker — main class
# ═══════════════════════════════════════════════════════════════════
class GazeTracker:
    """
    Real-time gaze tracker using MediaPipe Face Mesh iris landmarks.

    Features
    --------
    * Iris-center tracking (landmarks 468–477)
    * EMA (exponential moving average) smoothing
    * Blink detection via Eye Aspect Ratio (EAR)
    * Basic head-pose estimation (yaw / pitch)
    """

    def __init__(self, gaze_mapper=None):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=config.MAX_NUM_FACES,
            refine_landmarks=True,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self.gaze_mapper = gaze_mapper

        # EMA state
        self._smooth_x = None
        self._smooth_y = None
        self._alpha = config.EMA_ALPHA

        # Blink state
        self._blink_counter = 0
        self.total_blinks = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_frame(self, frame):
        """
        Process one BGR frame.  Returns a GazeData or None if no face found.
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark

        # ---- Iris centres ----
        left_iris  = self._iris_center(lm, config.LEFT_IRIS_IDX, w, h)
        right_iris = self._iris_center(lm, config.RIGHT_IRIS_IDX, w, h)
        iris_cx = (left_iris[0] + right_iris[0]) / 2
        iris_cy = (left_iris[1] + right_iris[1]) / 2

        # ---- EMA smoothing ----
        iris_cx, iris_cy = self._ema(iris_cx, iris_cy)

        # ---- Blink detection ----
        left_ear  = self._ear(lm, config.LEFT_EYE_EAR_IDX, w, h)
        right_ear = self._ear(lm, config.RIGHT_EYE_EAR_IDX, w, h)
        avg_ear = (left_ear + right_ear) / 2
        blink = avg_ear < config.EAR_THRESHOLD
        if blink:
            self._blink_counter += 1
        else:
            if self._blink_counter >= config.EAR_CONSEC_FRAMES:
                self.total_blinks += 1
            self._blink_counter = 0

        # ---- Head pose (simple yaw / pitch from key landmarks) ----
        head_yaw, head_pitch = self._head_pose(lm, w, h)

        # ---- Map to screen coordinates ----
        scr_x, scr_y = None, None
        if self.gaze_mapper is not None and not blink:
            scr_x, scr_y = self.gaze_mapper.map_to_screen(iris_cx, iris_cy)

        return GazeData(
            timestamp=datetime.now().timestamp(),
            left_iris=(left_iris[0], left_iris[1]),
            right_iris=(right_iris[0], right_iris[1]),
            iris_center=(iris_cx, iris_cy),
            gaze_screen_x=scr_x,
            gaze_screen_y=scr_y,
            left_ear=round(left_ear, 4),
            right_ear=round(right_ear, 4),
            blink=blink,
            head_yaw=round(head_yaw, 2),
            head_pitch=round(head_pitch, 2),
        )

    def draw_debug(self, frame, gaze):
        """Draw iris circles and info text onto frame (in-place)."""
        if gaze is None:
            cv2.putText(frame, "No face detected", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return

        # Iris dots
        for pt in (gaze.left_iris, gaze.right_iris):
            if pt:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        # Smoothed centre
        if gaze.iris_center:
            cv2.circle(frame, (int(gaze.iris_center[0]), int(gaze.iris_center[1])),
                       5, (255, 255, 0), -1)

        # EAR & blink
        ear_txt = f"EAR: {gaze.left_ear:.2f}/{gaze.right_ear:.2f}"
        blink_txt = f"Blinks: {self.total_blinks}"
        cv2.putText(frame, ear_txt, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, blink_txt, (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Head pose
        pose_txt = f"Yaw: {gaze.head_yaw:.1f}  Pitch: {gaze.head_pitch:.1f}"
        cv2.putText(frame, pose_txt, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        if gaze.blink:
            cv2.putText(frame, "BLINK", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def close(self):
        self.face_mesh.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _iris_center(landmarks, indices, w, h):
        xs, ys = [], []
        for idx in indices:
            lm = landmarks[idx]
            xs.append(lm.x * w)
            ys.append(lm.y * h)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _ema(self, x, y):
        if self._smooth_x is None:
            self._smooth_x, self._smooth_y = x, y
        else:
            a = self._alpha
            self._smooth_x = a * x + (1 - a) * self._smooth_x
            self._smooth_y = a * y + (1 - a) * self._smooth_y
        return self._smooth_x, self._smooth_y

    @staticmethod
    def _ear(landmarks, indices, w, h):
        """Compute Eye Aspect Ratio for blink detection."""
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
        # P1-P4 horizontal, P2-P6 & P3-P5 vertical
        hor = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        v1  = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2  = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        if hor == 0:
            return 0.0
        return (v1 + v2) / (2.0 * hor)

    @staticmethod
    def _head_pose(landmarks, w, h):
        """
        Rough yaw/pitch from landmark geometry.
        Yaw  = horizontal asymmetry of eye corners relative to nose.
        Pitch = vertical position of nose tip relative to forehead-chin midline.
        """
        def _pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        nose = _pt(config.NOSE_TIP_IDX)
        forehead = _pt(config.FOREHEAD_IDX)
        chin = _pt(config.CHIN_IDX)
        l_eye = _pt(config.LEFT_EYE_CORNER_IDX)
        r_eye = _pt(config.RIGHT_EYE_CORNER_IDX)

        # Yaw – ratio of nose-to-eye distances
        dl = np.linalg.norm(nose - l_eye)
        dr = np.linalg.norm(nose - r_eye)
        yaw = (dl - dr) / (dl + dr + 1e-6) * 90  # rough degrees

        # Pitch – nose position along forehead-chin axis
        fc = chin - forehead
        fn = nose - forehead
        fc_len = np.linalg.norm(fc) + 1e-6
        proj = np.dot(fn, fc) / fc_len
        ratio = proj / fc_len
        pitch = (ratio - 0.5) * 90  # rough degrees

        return yaw, pitch


# ═══════════════════════════════════════════════════════════════════
# Stand-alone runner (backwards-compatible with original tracking.py)
# ═══════════════════════════════════════════════════════════════════
def run_standalone():
    """
    Run eye tracking from webcam and save gaze data to CSV.
    Press Q to stop.
    """
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(config.RAW_DATA_DIR, f"gaze_{ts}.csv")

    # Optionally load calibration
    mapper = None
    try:
        from tracking.gaze_mapper import GazeMapper
        mapper = GazeMapper()
        print("✅ Calibration model loaded")
    except Exception:
        print("⚠ No calibration model found – recording raw iris data only")

    tracker = GazeTracker(gaze_mapper=mapper)
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open camera")

    csv_file = open(csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(GazeData.csv_header())

    print(f"📁 Saving gaze data to: {csv_path}")
    print("▶ Press Q to stop recording")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gaze = tracker.process_frame(frame)

        if gaze is not None:
            csv_writer.writerow(gaze.csv_row())

        tracker.draw_debug(frame, gaze)
        cv2.imshow("Eye Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    tracker.close()
    print(f"✅ Session saved – {tracker.total_blinks} blinks detected")


if __name__ == "__main__":
    run_standalone()
