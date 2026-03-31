"""
Gaze Mapper — loads the calibration model and converts raw iris
coordinates to estimated screen coordinates via an affine transform.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class GazeMapper:
    """
    Maps raw iris centre coordinates to screen coordinates
    using a pre-computed affine transform from calibration.

    Usage
    -----
        mapper = GazeMapper()                      # loads saved model
        sx, sy = mapper.map_to_screen(iris_x, iris_y)
    """

    def __init__(self, model_path=None):
        path = model_path or config.CALIB_MODEL_PATH

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Calibration model not found at '{path}'. "
                "Run calibration first:  python main.py calibrate"
            )

        data = np.load(path)
        self.M = data["affine_matrix"]          # (2, 3)
        self.iris_pts = data.get("iris_pts")    # (N, 2) — kept for debugging
        self.screen_pts = data.get("screen_pts")

    def map_to_screen(self, iris_x, iris_y):
        """
        Apply the affine transform to map iris coords → screen coords.
        Clamps the result to [0, SCREEN_W) × [0, SCREEN_H).
        """
        v = np.array([iris_x, iris_y, 1.0])
        out = self.M @ v  # (2,)
        sx = float(np.clip(out[0], 0, config.SCREEN_W - 1))
        sy = float(np.clip(out[1], 0, config.SCREEN_H - 1))
        return sx, sy

    def accuracy_report(self):
        """
        Return the mean error (pixels) between predicted and actual
        screen positions for the calibration points.
        """
        if self.iris_pts is None or self.screen_pts is None:
            return None
        errors = []
        for ip, sp in zip(self.iris_pts, self.screen_pts):
            pred = self.map_to_screen(ip[0], ip[1])
            err = np.linalg.norm(np.array(pred) - np.array(sp))
            errors.append(err)
        return {
            "mean_error_px": float(np.mean(errors)),
            "max_error_px": float(np.max(errors)),
            "per_point_errors": errors,
        }
