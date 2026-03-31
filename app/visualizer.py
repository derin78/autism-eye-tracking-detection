"""
Real-time gaze visualizer — draws the estimated gaze point as a
coloured dot with a fading trail on a semi-transparent fullscreen overlay.
"""

import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from tracking.tracking import GazeTracker
from tracking.gaze_mapper import GazeMapper


def run_visualizer():
    """
    Start the real-time gaze visualizer with a fullscreen overlay
    and camera debug view.
    """
    # Load mapper
    try:
        mapper = GazeMapper()
        print("✅ Calibration model loaded")
        report = mapper.accuracy_report()
        if report:
            print(f"   Mean calibration error: {report['mean_error_px']:.1f} px")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Please run calibration first:  python main.py calibrate")
        return

    tracker = GazeTracker(gaze_mapper=mapper)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open camera")

    # Fullscreen overlay window
    overlay_win = "Gaze Overlay"
    cv2.namedWindow(overlay_win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(overlay_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Camera debug window
    cam_win = "Camera Feed"

    trail = deque(maxlen=config.GAZE_TRAIL_LENGTH)
    fps_time = time.time()
    fps_counter = 0
    fps_display = 0

    print("▶ Press Q to stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        gaze = tracker.process_frame(frame)
        tracker.draw_debug(frame, gaze)

        # ---- Build overlay ----
        canvas = np.zeros((config.SCREEN_H, config.SCREEN_W, 3), dtype=np.uint8)

        if gaze and gaze.gaze_screen_x is not None and not gaze.blink:
            sx, sy = int(gaze.gaze_screen_x), int(gaze.gaze_screen_y)
            trail.append((sx, sy))

            # Draw trail (fading)
            for i, (tx, ty) in enumerate(trail):
                alpha = (i + 1) / len(trail)
                radius = max(3, int(config.GAZE_DOT_RADIUS * alpha * 0.6))
                color = tuple(int(c * alpha) for c in config.GAZE_DOT_COLOR)
                cv2.circle(canvas, (tx, ty), radius, color, -1)

            # Main gaze dot
            cv2.circle(canvas, (sx, sy), config.GAZE_DOT_RADIUS,
                       config.GAZE_DOT_COLOR, -1)
            cv2.circle(canvas, (sx, sy), config.GAZE_DOT_RADIUS + 4,
                       (255, 255, 255), 2)

        # ---- Info panel ----
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_time = time.time()

        info_lines = [
            f"FPS: {fps_display}",
            f"Blinks: {tracker.total_blinks}",
        ]
        if gaze:
            info_lines.append(f"EAR: {gaze.left_ear:.2f} / {gaze.right_ear:.2f}")
            if gaze.gaze_screen_x is not None:
                info_lines.append(
                    f"Gaze: ({gaze.gaze_screen_x:.0f}, {gaze.gaze_screen_y:.0f})"
                )
            status = "BLINK" if gaze.blink else "TRACKING"
        else:
            status = "NO FACE"
        info_lines.insert(0, status)

        y0 = 30
        for i, line in enumerate(info_lines):
            color = (0, 0, 255) if i == 0 and status != "TRACKING" else (200, 200, 200)
            cv2.putText(canvas, line, (20, y0 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)

        cv2.imshow(overlay_win, canvas)
        cv2.imshow(cam_win, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("✅ Visualizer stopped")


if __name__ == "__main__":
    run_visualizer()
