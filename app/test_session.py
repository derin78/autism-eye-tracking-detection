"""
Test session — shows stimulus images fullscreen while tracking gaze,
then calculates face-attention metrics per image.
"""

import csv
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from tracking.tracking import GazeTracker


def _load_stimuli():
    """Load all stimulus images and return as list of (image, face_side)."""
    images = []
    for i in range(config.STIMULUS_COUNT):
        path = os.path.join(config.STIMULI_DIR, f"stimulus_{i+1:02d}.png")
        if not os.path.exists(path):
            print(f"⚠ Missing stimulus: {path}")
            continue
        img = cv2.imread(path)
        if img is not None:
            side = config.FACE_SIDE_LABELS[i] if i < len(config.FACE_SIDE_LABELS) else "left"
            images.append((img, side, i + 1))
    return images


def run_test_session(on_progress=None):
    """
    Run a complete autism screening test session.

    Parameters
    ----------
    on_progress : callable or None
        Optional callback(step, total, message) for GUI progress updates.

    Returns
    -------
    dict with results: per_image metrics, overall scores, raw data path
    """
    stimuli = _load_stimuli()
    if not stimuli:
        raise FileNotFoundError(
            "No stimulus images found. Run: python stimuli/generate_stimuli.py"
        )

    # ── Setup tracker ──
    tracker = GazeTracker()
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    # Prepare output
    os.makedirs(config.TEST_RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_csv_path = os.path.join(config.TEST_RESULTS_DIR, f"test_{ts}.csv")
    csv_file = open(raw_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "stimulus_id", "timestamp",
        "iris_center_x", "iris_center_y",
        "face_side", "looking_at_face",
    ])

    # ── Create fullscreen window ──
    win = "Autism Screening Test"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    per_image_results = []

    # ── Show "Get Ready" screen ──
    ready = np.zeros((config.SCREEN_H, config.SCREEN_W, 3), dtype=np.uint8)
    cv2.putText(ready, "Autism Eye Tracking Test",
                (config.SCREEN_W // 2 - 280, config.SCREEN_H // 2 - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(ready, "Look naturally at each image that appears.",
                (config.SCREEN_W // 2 - 280, config.SCREEN_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
    cv2.putText(ready, "Press SPACE to begin...",
                (config.SCREEN_W // 2 - 160, config.SCREEN_H // 2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 100), 1)
    cv2.imshow(win, ready)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 32:  # SPACE
            break
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            csv_file.close()
            tracker.close()
            return None
        # Keep reading camera to keep tracker warm
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            tracker.process_frame(frame)

    # ── Show each stimulus ──
    total = len(stimuli)
    for idx, (stim_img, face_side, stim_id) in enumerate(stimuli):
        if on_progress:
            on_progress(idx + 1, total, f"Showing stimulus {idx + 1}/{total}")

        # Resize stimulus to screen size
        display_img = cv2.resize(stim_img, (config.SCREEN_W, config.SCREEN_H))
        half_w = config.SCREEN_W // 2

        face_samples = 0
        object_samples = 0
        all_samples = 0

        start_time = time.time()
        while time.time() - start_time < config.STIMULUS_DISPLAY_SEC:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            gaze = tracker.process_frame(frame)

            if gaze and gaze.iris_center and not gaze.blink:
                iris_x = gaze.iris_center[0]
                frame_w = frame.shape[1]
                all_samples += 1

                # Determine which side the user is looking at
                # iris_x is in camera coords: left of camera = right of screen (mirrored)
                # Normalize to screen position roughly
                norm_x = iris_x / frame_w  # 0 = left of screen, 1 = right

                if face_side == "left":
                    looking_at_face = norm_x < 0.5
                else:
                    looking_at_face = norm_x >= 0.5

                if looking_at_face:
                    face_samples += 1
                else:
                    object_samples += 1

                csv_writer.writerow([
                    stim_id, time.time(),
                    gaze.iris_center[0], gaze.iris_center[1],
                    face_side, int(looking_at_face),
                ])

            # Show stimulus with progress bar
            canvas = display_img.copy()
            elapsed = time.time() - start_time
            progress = elapsed / config.STIMULUS_DISPLAY_SEC
            bar_y = config.SCREEN_H - 8
            cv2.rectangle(canvas, (0, bar_y),
                          (int(config.SCREEN_W * progress), config.SCREEN_H),
                          (0, 200, 100), -1)

            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Calculate face attention ratio for this image
        face_ratio = face_samples / max(all_samples, 1)
        per_image_results.append({
            "stimulus_id": stim_id,
            "face_side": face_side,
            "total_samples": all_samples,
            "face_samples": face_samples,
            "object_samples": object_samples,
            "face_attention_ratio": round(face_ratio, 4),
        })

        # Brief pause between stimuli
        blank = np.zeros((config.SCREEN_H, config.SCREEN_W, 3), dtype=np.uint8)
        cv2.putText(blank, "+",
                    (config.SCREEN_W // 2 - 15, config.SCREEN_H // 2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)
        cv2.imshow(win, blank)
        cv2.waitKey(1000)  # 1 second fixation cross

    # ── Done — clean up OpenCV fully ──
    try:
        cv2.destroyWindow(win)
    except Exception:
        pass
    cv2.destroyAllWindows()
    cv2.waitKey(1)   # flush remaining events
    cap.release()
    csv_file.close()
    tracker.close()
    time.sleep(0.3)  # let OS fully release the window

    # Overall metrics
    if per_image_results:
        ratios = [r["face_attention_ratio"] for r in per_image_results]
        overall_face_ratio = sum(ratios) / len(ratios)
    else:
        overall_face_ratio = 0

    results = {
        "timestamp": ts,
        "per_image": per_image_results,
        "overall_face_attention_ratio": round(overall_face_ratio, 4),
        "total_blinks": tracker.total_blinks,
        "raw_data_path": raw_csv_path,
    }

    if on_progress:
        on_progress(total, total, "Test complete!")

    return results


if __name__ == "__main__":
    results = run_test_session()
    if results:
        print(f"\n✅ Test completed!")
        print(f"   Overall face attention: {results['overall_face_attention_ratio']:.1%}")
        print(f"   Blinks: {results['total_blinks']}")
        print(f"   Raw data: {results['raw_data_path']}")
