"""
Central configuration for the Autism Eye Tracking System.
All tunable parameters are defined here.
"""

import os

# =============================================================================
# Screen dimensions
# =============================================================================
# Try auto-detect; fall back to common defaults
try:
    from screeninfo import get_monitors
    _m = get_monitors()[0]
    SCREEN_W, SCREEN_H = _m.width, _m.height
except Exception:
    SCREEN_W, SCREEN_H = 1920, 1080

# =============================================================================
# MediaPipe
# =============================================================================
MAX_NUM_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Iris landmark indices (MediaPipe Face Mesh with refine_landmarks=True)
LEFT_IRIS_IDX = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]

# Eye contour landmarks for EAR (Eye Aspect Ratio) blink detection
# Each list: [P1, P2, P3, P4, P5, P6] – top/bottom pairs for vertical dist
LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

# Head-pose reference landmarks
NOSE_TIP_IDX = 1
FOREHEAD_IDX = 10
CHIN_IDX = 152
LEFT_EYE_CORNER_IDX = 33
RIGHT_EYE_CORNER_IDX = 263
LEFT_MOUTH_IDX = 61
RIGHT_MOUTH_IDX = 291

# =============================================================================
# Smoothing
# =============================================================================
EMA_ALPHA = 0.4          # Exponential moving average factor (0–1, lower = more smooth)

# =============================================================================
# Blink detection
# =============================================================================
EAR_THRESHOLD = 0.21     # Below this EAR → eye is closed
EAR_CONSEC_FRAMES = 2    # Consecutive frames below threshold to count as blink

# =============================================================================
# Calibration
# =============================================================================
CALIB_POINTS_ROWS = 3    # 3×3 = 9-point calibration (configurable)
CALIB_POINTS_COLS = 3
CALIB_MARGIN = 100        # Pixels from screen edge
CALIB_HOLD_SEC = 2.0      # Seconds to stare at each point
CALIB_COUNTDOWN_SEC = 1.0  # Countdown before recording starts
CALIB_MODEL_PATH = os.path.join("data", "calibration_model.npz")

# =============================================================================
# Data paths
# =============================================================================
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
FEATURES_DATA_DIR = os.path.join("data", "features")
EXPERIMENTS_DIR = "experiments"

# =============================================================================
# Data processing
# =============================================================================
SAVGOL_WINDOW = 15        # Savitzky-Golay filter window length (must be odd)
SAVGOL_POLY = 3           # Polynomial order

# Fixation / saccade detection (velocity-based, degrees/sec if calibrated)
FIXATION_VEL_THRESHOLD = 30   # pixels/sec – below = fixation
SACCADE_VEL_THRESHOLD = 100   # pixels/sec – above = saccade

# =============================================================================
# Visualisation
# =============================================================================
GAZE_DOT_RADIUS = 18
GAZE_DOT_COLOR = (0, 255, 100)   # BGR - green
GAZE_TRAIL_LENGTH = 20
OVERLAY_ALPHA = 0.35

# =============================================================================
# Camera
# =============================================================================
CAMERA_PREVIEW_W = 640
CAMERA_PREVIEW_H = 480
CAMERA_INDEX = 0  # 0 = laptop, 1 = phone/external, etc.

# =============================================================================
# Stimulus test
# =============================================================================
STIMULI_DIR = "stimuli"
STIMULUS_DISPLAY_SEC = 5.0      # How long each image is shown
STIMULUS_COUNT = 10             # Number of test images
FACE_SIDE_LABELS = [            # Which side has the face in each stimulus
    "left", "right", "left", "right", "left",
    "right", "left", "right", "left", "right",
]

# =============================================================================
# Test results & reports
# =============================================================================
TEST_RESULTS_DIR = os.path.join("data", "test_results")
REPORTS_DIR = os.path.join("experiments", "reports")

# Classification thresholds (based on published research)
# Typical observers spend ~65-80% of time on faces
FACE_ATTENTION_TYPICAL_MIN = 0.55   # Above this = more typical
RISK_THRESHOLD_HIGH = 0.35          # Below this face ratio = high risk
RISK_THRESHOLD_MODERATE = 0.50      # Below this = moderate risk

