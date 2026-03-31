"""
Data processing — load raw gaze CSVs, clean, smooth, and segment
into fixations and saccades.
"""

import os
import sys
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None


def normalize_columns(df):
    """
    Map old-format CSV columns to the new standard names so the
    pipeline works with data recorded by the original tracking.py.

    Old format:  left_eye_x/y, right_eye_x/y, raw_gaze_x/y
    New format:  iris_center_x/y, gaze_screen_x/y, etc.
    """
    # If already in new format, nothing to do
    if "iris_center_x" in df.columns:
        return df

    # Compute iris centre from left/right eye columns
    if "left_eye_x" in df.columns and "right_eye_x" in df.columns:
        df["iris_center_x"] = (df["left_eye_x"] + df["right_eye_x"]) / 2
        df["iris_center_y"] = (df["left_eye_y"] + df["right_eye_y"]) / 2

    # Rename raw_gaze to gaze_screen if present
    rename_map = {}
    if "raw_gaze_x" in df.columns:
        rename_map["raw_gaze_x"] = "gaze_screen_x"
    if "raw_gaze_y" in df.columns:
        rename_map["raw_gaze_y"] = "gaze_screen_y"
    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def load_latest_raw(directory=None):
    """Load the most recent raw gaze CSV and return as DataFrame."""
    d = directory or config.RAW_DATA_DIR
    files = sorted(glob.glob(os.path.join(d, "gaze_*.csv")))
    if not files:
        raise FileNotFoundError(f"No raw CSV files found in {d}")
    path = files[-1]
    print(f"📂 Loading {path}")
    return pd.read_csv(path), path


def clean(df):
    """Remove blink frames and rows with missing data."""
    if "blink" in df.columns:
        df = df[df["blink"] != 1].copy()
    # Only filter on iris_center if available
    if "iris_center_x" in df.columns:
        df = df.dropna(subset=["iris_center_x", "iris_center_y"])
    df = df.reset_index(drop=True)
    return df


def smooth(df, cols=("iris_center_x", "iris_center_y",
                     "gaze_screen_x", "gaze_screen_y")):
    """Apply Savitzky-Golay filter for offline smoothing."""
    if savgol_filter is None:
        print("⚠ scipy not available, skipping Savitzky-Golay smoothing")
        return df

    win = config.SAVGOL_WINDOW
    poly = config.SAVGOL_POLY

    for col in cols:
        if col in df.columns and df[col].notna().sum() > win:
            df[col] = savgol_filter(df[col].values, win, poly)
    return df


def compute_velocity(df, x_col="iris_center_x", y_col="iris_center_y"):
    """Compute point-to-point velocity in pixels/second."""
    dx = df[x_col].diff()
    dy = df[y_col].diff()
    dt = df["timestamp"].diff()
    dt = dt.replace(0, np.nan)
    speed = np.sqrt(dx**2 + dy**2) / dt
    df["velocity"] = speed
    return df


def segment_fixations_saccades(df):
    """
    Label each sample as 'fixation', 'saccade', or 'other'
    based on velocity thresholds.
    """
    if "velocity" not in df.columns:
        df = compute_velocity(df)

    labels = []
    for v in df["velocity"]:
        if pd.isna(v):
            labels.append("other")
        elif v < config.FIXATION_VEL_THRESHOLD:
            labels.append("fixation")
        elif v > config.SACCADE_VEL_THRESHOLD:
            labels.append("saccade")
        else:
            labels.append("other")
    df["event"] = labels
    return df


def process_raw(raw_path=None):
    """
    Full pipeline: load → clean → smooth → velocity → segment → save.
    Returns processed DataFrame and output path.
    """
    if raw_path:
        df = pd.read_csv(raw_path)
        basename = os.path.basename(raw_path)
    else:
        df, raw_path = load_latest_raw()
        basename = os.path.basename(raw_path)

    df = normalize_columns(df)

    print(f"  Raw samples: {len(df)}")
    df = clean(df)
    print(f"  After cleaning: {len(df)}")
    df = smooth(df)
    df = compute_velocity(df)
    df = segment_fixations_saccades(df)

    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    out_name = basename.replace("gaze_", "processed_")
    out_path = os.path.join(config.PROCESSED_DATA_DIR, out_name)
    df.to_csv(out_path, index=False)
    print(f"✅ Processed data saved to {out_path}")
    return df, out_path


if __name__ == "__main__":
    process_raw()
