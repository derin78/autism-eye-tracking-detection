"""
Fullscreen 9-point calibration with visual countdown.
Collects iris positions while the user stares at each point,
then fits an affine transform and saves it to disk.
"""

import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ═══════════════════════════════════════════════════════════════════
# Helper — iris centre (same logic as GazeTracker)
# ═══════════════════════════════════════════════════════════════════
def _iris_center(landmarks, indices, w, h):
    xs, ys = [], []
    for idx in indices:
        lm = landmarks[idx]
        xs.append(lm.x * w)
        ys.append(lm.y * h)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


# ═══════════════════════════════════════════════════════════════════
# Generate calibration grid
# ═══════════════════════════════════════════════════════════════════
def _calibration_points():
    """Return a list of (screen_x, screen_y) calibration targets."""
    rows, cols = config.CALIB_POINTS_ROWS, config.CALIB_POINTS_COLS
    m = config.CALIB_MARGIN
    points = []
    for r in range(rows):
        for c in range(cols):
            x = m + c * (config.SCREEN_W - 2 * m) / max(cols - 1, 1)
            y = m + r * (config.SCREEN_H - 2 * m) / max(rows - 1, 1)
            points.append((int(x), int(y)))
    return points


# ═══════════════════════════════════════════════════════════════════
# Fit affine transform
# ═══════════════════════════════════════════════════════════════════
def _fit_affine(iris_pts, screen_pts):
    """
    Fit affine transform: screen = M @ [iris_x, iris_y, 1]^T

    iris_pts   – (N, 2) array of averaged iris positions
    screen_pts – (N, 2) array of known screen points

    Returns M (2×3 matrix).
    """
    N = len(iris_pts)
    # Build A matrix [iris_x, iris_y, 1]
    A = np.hstack([np.array(iris_pts), np.ones((N, 1))])
    B = np.array(screen_pts)

    # Solve using least-squares  (B = A @ M^T)
    M_T, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return M_T.T  # (2, 3)


# ═══════════════════════════════════════════════════════════════════
# Main calibration routine
# ═══════════════════════════════════════════════════════════════════
def run_calibration():
    """
    Run a fullscreen calibration session:
    1. Show a dot at each calibration point
    2. Collect iris samples while the user looks at the dot
    3. Fit an affine transform and save to disk
    """
    points = _calibration_points()

    # MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open camera")

    # Fullscreen window
    win = "Calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    iris_avg = []     # averaged iris positions per point
    screen_pts = []   # corresponding screen positions

    for idx, point in enumerate(points):
        samples = []

        # ---- Countdown phase ----
        cd_start = time.time()
        while time.time() - cd_start < config.CALIB_COUNTDOWN_SEC:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            canvas = np.zeros((config.SCREEN_H, config.SCREEN_W, 3), dtype=np.uint8)

            # Point marker (pulsing ring)
            elapsed = time.time() - cd_start
            radius = int(20 + 10 * np.sin(elapsed * 6))
            cv2.circle(canvas, point, radius, (0, 100, 255), 3)

            # Info text
            remaining = max(0, config.CALIB_COUNTDOWN_SEC - elapsed)
            cv2.putText(canvas, f"Point {idx+1}/{len(points)} — look at the dot",
                        (50, config.SCREEN_H - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(canvas, f"Starting in {remaining:.1f}s",
                        (50, config.SCREEN_H - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to abort
                cap.release()
                cv2.destroyAllWindows()
                face_mesh.close()
                print("❌ Calibration aborted")
                return

        # ---- Sample collection phase ----
        rec_start = time.time()
        while time.time() - rec_start < config.CALIB_HOLD_SEC:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            canvas = np.zeros((config.SCREEN_H, config.SCREEN_W, 3), dtype=np.uint8)

            # Solid recording dot
            cv2.circle(canvas, point, 14, (0, 255, 100), -1)
            cv2.circle(canvas, point, 18, (0, 255, 100), 2)

            # Progress bar
            progress = (time.time() - rec_start) / config.CALIB_HOLD_SEC
            bar_w = 300
            cv2.rectangle(canvas, (50, config.SCREEN_H - 30),
                          (50 + int(bar_w * progress), config.SCREEN_H - 15),
                          (0, 255, 100), -1)
            cv2.rectangle(canvas, (50, config.SCREEN_H - 30),
                          (50 + bar_w, config.SCREEN_H - 15),
                          (100, 100, 100), 1)

            cv2.putText(canvas, f"Recording point {idx+1}/{len(points)}...",
                        (50, config.SCREEN_H - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 1)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                li = _iris_center(lm, config.LEFT_IRIS_IDX, w, h)
                ri = _iris_center(lm, config.RIGHT_IRIS_IDX, w, h)
                cx = (li[0] + ri[0]) / 2
                cy = (li[1] + ri[1]) / 2
                samples.append((cx, cy))

            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                face_mesh.close()
                print("❌ Calibration aborted")
                return

        # Average samples for this point
        if samples:
            avg_x = sum(s[0] for s in samples) / len(samples)
            avg_y = sum(s[1] for s in samples) / len(samples)
            iris_avg.append((avg_x, avg_y))
            screen_pts.append(point)
            print(f"  ✅ Point {idx+1}: {len(samples)} samples → iris ({avg_x:.1f}, {avg_y:.1f})")
        else:
            print(f"  ⚠ Point {idx+1}: no samples collected (face not detected)")

    cap.release()
    try:
        cv2.destroyWindow(win)
    except Exception:
        pass
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    face_mesh.close()
    time.sleep(0.3)

    # ---- Fit and save ----
    if len(iris_avg) < 3:
        print("❌ Not enough points to compute calibration (need ≥3)")
        return

    M = _fit_affine(iris_avg, screen_pts)
    os.makedirs(os.path.dirname(config.CALIB_MODEL_PATH), exist_ok=True)
    np.savez(config.CALIB_MODEL_PATH, affine_matrix=M,
             iris_pts=np.array(iris_avg),
             screen_pts=np.array(screen_pts))
    print(f"✅ Calibration saved to {config.CALIB_MODEL_PATH}")
    print(f"   Affine matrix:\n{M}")


if __name__ == "__main__":
    run_calibration()
