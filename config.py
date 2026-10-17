"""
Configuration file for AI-Based Driver Drowsiness Detection System (DrowsiGuard).
Defines eye and mouth landmark indices, default thresholds, mock data, and UI constants.
"""

import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Directory Paths
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# File Paths
ALARM_WAV_PATH = BASE_DIR / "alarm.wav"
HISTORY_CSV_PATH = DATA_DIR / "history.csv"
INCIDENTS_CSV_PATH = DATA_DIR / "incidents.csv"
DRIVERS_CSV_PATH = DATA_DIR / "drivers.csv"
LOGO_PATH = ASSETS_DIR / "logo.png"
BANNER_PATH = ASSETS_DIR / "banner.png"
MODEL_PATH = MODELS_DIR / "face_landmarker.task"

# Ensure required directories exist
for directory in [ASSETS_DIR, DATA_DIR, SCREENSHOTS_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# MediaPipe Face Mesh Landmark Indices
# Right Eye (User's Right / Camera's Left)
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

# Left Eye (User's Left / Camera's Right)
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Mouth Landmarks for Yawn / MAR Detection (Upper lip, Lower lip, Left corner, Right corner)
MOUTH_LANDMARKS = [13, 14, 61, 291, 82, 87, 312, 317]

# Key 3D Head Pose Landmark Indices (Nose, Chin, Left Eye Corner, Right Eye Corner, Left Mouth Corner, Right Mouth Corner)
HEAD_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

# Default Algorithm Thresholds
DEFAULT_EAR_THRESHOLD = 0.25
DEFAULT_CONSEC_FRAMES = 16
DEFAULT_EYE_CLOSURE_TIME = 2.0  # seconds
DEFAULT_YAWN_MAR_THRESHOLD = 0.40
DEFAULT_YAWN_LIMIT = 3          # yawns in 10 mins
DEFAULT_HEAD_TILT_ANGLE = 25.0  # degrees
DEFAULT_HEAD_TILT_DURATION = 3.0 # seconds
DEFAULT_ESCALATION_WINDOW = 10  # seconds

DEFAULT_WARNING_FATIGUE_SCORE = 0.50
DEFAULT_CRITICAL_FATIGUE_SCORE = 0.75

# Threshold Allowed Ranges
THRESHOLD_RANGES = {
    "eye_closure_time": (0.5, 5.0),
    "yawn_count": (1, 10),
    "head_pose_angle": (10, 45),
    "fatigue_score_warning": (0.1, 1.0),
    "fatigue_score_critical": (0.1, 1.0),
    "escalation_window": (3, 30),
}

# Maximum history points to store for real-time Plotly charts
MAX_CHART_HISTORY = 100

# UI Theme Color Palettes
COLORS = {
    "ALERT": (0, 0, 255),      # BGR for Red warning box
    "AWAKE": (0, 255, 0),      # BGR for Green alert box
    "LANDMARK": (255, 200, 0), # BGR for Eye landmarks (Cyan/Yellow)
    "TEXT": (255, 255, 255),   # BGR for Text overlay
    "PRIMARY_HEX": "#00B4D8",  # Cyan/Blue primary
    "ACCENT_HEX": "#10B981",   # Emerald green accent
    "DANGER_HEX": "#EF4444",   # Red danger theme
    "WARNING_HEX": "#F59E0B",  # Amber warning theme
    "DARK_BG": "#0B132B",      # Deep dark navy for edge unit
    "CARD_BG": "#1C2541",      # Card background dark
}

# Default Active Driver & Vehicle Baseline
DEFAULT_DRIVER = {
    "driver_id": "DRV2026A",
    "name": "Rahul Verma",
    "email": "rahul.verma@fleet.in",
    "phone": "9876543210",
    "vehicle_id": "UP32-AB-1234",
    "licence_no": "UP32 20190012345",
    "dob": "14 / 08 / 1994",
    "shift": "Night (10 PM – 6 AM)",
    "calibration_status": "Completed",
    "baseline_ear": 0.31,
    "resting_hr": 70
}
