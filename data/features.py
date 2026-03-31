"""
Feature extraction from processed gaze data.
Extracts fixation, saccade, and scanpath metrics relevant to
autism-related gaze pattern analysis.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def extract_features(df):
    """
    Extract gaze features from a processed DataFrame.
    Returns a dict of feature values.
    """
    features = {}
    duration = 0
    if "timestamp" in df.columns and len(df) > 1:
        duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    features["session_duration_sec"] = round(duration, 2)
    features["total_samples"] = len(df)

    # ── Fixation metrics ──────────────────────────────────────────
    fix = df[df["event"] == "fixation"] if "event" in df.columns else pd.DataFrame()
    features["fixation_sample_count"] = len(fix)
    features["fixation_ratio"] = round(len(fix) / max(len(df), 1), 4)

    if len(fix) > 1:
        # Group consecutive fixation samples into fixation episodes
        fix = fix.copy()
        fix["group"] = (fix.index.to_series().diff() > 1).cumsum()
        groups = fix.groupby("group")

        durations = groups["timestamp"].apply(lambda t: t.iloc[-1] - t.iloc[0])
        features["fixation_count"] = len(groups)
        features["fixation_mean_duration_sec"] = round(durations.mean(), 4)
        features["fixation_max_duration_sec"] = round(durations.max(), 4)
        features["fixation_total_duration_sec"] = round(durations.sum(), 4)

        # Spatial spread of fixation centres
        centres_x = groups["iris_center_x"].mean()
        centres_y = groups["iris_center_y"].mean()
        features["fixation_spatial_std_x"] = round(centres_x.std(), 2)
        features["fixation_spatial_std_y"] = round(centres_y.std(), 2)
    else:
        features["fixation_count"] = 0
        features["fixation_mean_duration_sec"] = 0
        features["fixation_max_duration_sec"] = 0
        features["fixation_total_duration_sec"] = 0
        features["fixation_spatial_std_x"] = 0
        features["fixation_spatial_std_y"] = 0

    # ── Saccade metrics ───────────────────────────────────────────
    sac = df[df["event"] == "saccade"] if "event" in df.columns else pd.DataFrame()
    features["saccade_sample_count"] = len(sac)
    features["saccade_ratio"] = round(len(sac) / max(len(df), 1), 4)

    if len(sac) > 1:
        sac = sac.copy()
        sac["group"] = (sac.index.to_series().diff() > 1).cumsum()
        groups = sac.groupby("group")

        features["saccade_count"] = len(groups)
        if "velocity" in sac.columns:
            features["saccade_mean_velocity"] = round(sac["velocity"].mean(), 2)
            features["saccade_peak_velocity"] = round(sac["velocity"].max(), 2)

        # Amplitude – distance between start and end of each saccade
        amplitudes = []
        for _, g in groups:
            dx = g["iris_center_x"].iloc[-1] - g["iris_center_x"].iloc[0]
            dy = g["iris_center_y"].iloc[-1] - g["iris_center_y"].iloc[0]
            amplitudes.append(np.sqrt(dx**2 + dy**2))
        features["saccade_mean_amplitude_px"] = round(np.mean(amplitudes), 2)
    else:
        features["saccade_count"] = 0
        features["saccade_mean_velocity"] = 0
        features["saccade_peak_velocity"] = 0
        features["saccade_mean_amplitude_px"] = 0

    # ── Scanpath metrics ──────────────────────────────────────────
    if "iris_center_x" in df.columns and len(df) > 1:
        dx = df["iris_center_x"].diff().dropna()
        dy = df["iris_center_y"].diff().dropna()
        path_length = np.sqrt(dx**2 + dy**2).sum()
        features["scanpath_length_px"] = round(path_length, 2)

        # Scanpath ratio = direct distance / path length
        start = df[["iris_center_x", "iris_center_y"]].iloc[0].values
        end = df[["iris_center_x", "iris_center_y"]].iloc[-1].values
        direct = np.linalg.norm(end - start)
        features["scanpath_ratio"] = round(
            direct / max(path_length, 1e-6), 4
        )
    else:
        features["scanpath_length_px"] = 0
        features["scanpath_ratio"] = 0

    # ── Blink metrics ─────────────────────────────────────────────
    if "blink" in df.columns:
        blink_rows = df[df["blink"] == 1]
        features["blink_frame_count"] = len(blink_rows)

    return features


def extract_and_save(processed_path):
    """
    Extract features from a processed CSV and save to features dir.
    """
    df = pd.read_csv(processed_path)
    feats = extract_features(df)

    os.makedirs(config.FEATURES_DATA_DIR, exist_ok=True)
    basename = os.path.basename(processed_path).replace("processed_", "features_")
    out_path = os.path.join(config.FEATURES_DATA_DIR, basename)

    feat_df = pd.DataFrame([feats])
    feat_df.to_csv(out_path, index=False)
    print(f"✅ Features saved to {out_path}")

    # Pretty print
    print("\n📊 Extracted Features:")
    print("─" * 45)
    for k, v in feats.items():
        print(f"  {k:40s} {v}")
    print("─" * 45)

    return feats, out_path


if __name__ == "__main__":
    import glob
    files = sorted(glob.glob(os.path.join(config.PROCESSED_DATA_DIR, "processed_*.csv")))
    if files:
        extract_and_save(files[-1])
    else:
        print("No processed files found. Run processing first.")
