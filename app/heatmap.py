"""
Heatmap generator — creates a 2D Gaussian heatmap from recorded
gaze data and saves it as a PNG image.
"""

import os
import sys
import glob

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

try:
    import pandas as pd
except ImportError:
    pd = None


def generate_heatmap(csv_path=None, output_path=None,
                     sigma=40, use_screen_coords=True):
    """
    Generate a gaze heatmap from a processed (or raw) CSV file.

    Parameters
    ----------
    csv_path : str or None
        Path to gaze CSV. If None, uses latest processed file.
    output_path : str or None
        Where to save the heatmap PNG. If None, auto-generates path.
    sigma : int
        Gaussian blur radius (higher = more diffuse heat blobs).
    use_screen_coords : bool
        If True, use gaze_screen_x/y; else use iris_center_x/y.
    """
    if pd is None:
        raise ImportError("pandas is required for heatmap generation")

    # ---- Load data ----
    if csv_path is None:
        search_dirs = [config.PROCESSED_DATA_DIR, config.RAW_DATA_DIR]
        for d in search_dirs:
            files = sorted(glob.glob(os.path.join(d, "*.csv")))
            if files:
                csv_path = files[-1]
                break
        if csv_path is None:
            raise FileNotFoundError("No gaze CSV files found")

    print(f"📂 Loading {csv_path}")
    df = pd.read_csv(csv_path)

    # ---- Choose coordinate columns ----
    if use_screen_coords and "gaze_screen_x" in df.columns:
        x_col, y_col = "gaze_screen_x", "gaze_screen_y"
        w, h = config.SCREEN_W, config.SCREEN_H
    elif "iris_center_x" in df.columns:
        x_col, y_col = "iris_center_x", "iris_center_y"
        # For raw iris data, use camera resolution as canvas
        w, h = 640, 480
    else:
        # Fallback to old-format columns
        x_col, y_col = "left_eye_x", "left_eye_y"
        w, h = 640, 480

    df = df.dropna(subset=[x_col, y_col])
    if "blink" in df.columns:
        df = df[df["blink"] != 1]

    xs = df[x_col].values
    ys = df[y_col].values

    print(f"  Using columns: {x_col}, {y_col}")
    print(f"  Canvas size: {w}×{h}")
    print(f"  Data points: {len(xs)}")

    # ---- Build density map ----
    heatmap = np.zeros((h, w), dtype=np.float64)
    for x, y in zip(xs, ys):
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            heatmap[iy, ix] += 1

    # ---- Gaussian blur ----
    if gaussian_filter is not None:
        heatmap = gaussian_filter(heatmap, sigma=sigma)
    else:
        # Fallback: OpenCV GaussianBlur
        ksize = sigma * 4 + 1  # ensure odd
        if ksize % 2 == 0:
            ksize += 1
        heatmap = cv2.GaussianBlur(heatmap, (ksize, ksize), sigma)

    # ---- Normalize & colorize ----
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Dark overlay for low-activity areas
    mask = heatmap_uint8 < 10
    colored[mask] = [20, 20, 20]

    # ---- Save ----
    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    if output_path is None:
        basename = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(config.EXPERIMENTS_DIR, f"heatmap_{basename}.png")

    cv2.imwrite(output_path, colored)
    print(f"✅ Heatmap saved to {output_path}")

    return output_path


if __name__ == "__main__":
    generate_heatmap()
