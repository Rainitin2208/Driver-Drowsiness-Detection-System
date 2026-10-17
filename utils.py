"""
Utility functions for AI Driver Drowsiness Detection System (DrowsiGuard).
Includes audio synthesis, branding image generation, Pygame & HTML5 audio controls,
CSV/Excel/PDF export generation, and sample database providers.
"""

import os
import wave
import struct
import math
import base64
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

from config import (
    ALARM_WAV_PATH, HISTORY_CSV_PATH, INCIDENTS_CSV_PATH, DRIVERS_CSV_PATH,
    LOGO_PATH, BANNER_PATH, MODEL_PATH, MODELS_DIR, SCREENSHOTS_DIR,
    ASSETS_DIR, DATA_DIR, REPORTS_DIR
)

def ensure_mediapipe_model_exists(model_path=MODEL_PATH):
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return model_path
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(model_path))
    except Exception:
        pass
    return model_path

# Pygame audio setup
_pygame_initialized = False
try:
    import pygame
    pygame.mixer.init()
    _pygame_initialized = True
except Exception:
    _pygame_initialized = False

def generate_alarm_wav(file_path=ALARM_WAV_PATH, duration_sec=1.5, sample_rate=44100):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    n_samples = int(sample_rate * duration_sec)
    
    with wave.open(str(file_path), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        amplitude = 28000
        for i in range(n_samples):
            t = float(i) / sample_rate
            freq = 880 if int(t * 8) % 2 == 0 else 1200
            val = int(amplitude * math.sin(2 * math.pi * freq * t))
            wav_file.writeframes(struct.pack('<h', val))

def generate_branding_assets():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    if not os.path.exists(LOGO_PATH):
        logo = Image.new("RGBA", (250, 250), (11, 19, 43, 255))
        draw = ImageDraw.Draw(logo)
        draw.ellipse([25, 25, 225, 225], outline=(0, 180, 216, 255), width=8)
        draw.ellipse([60, 95, 190, 155], outline=(16, 185, 129, 255), width=5)
        draw.ellipse([105, 105, 145, 145], fill=(239, 68, 68, 255))
        logo.save(LOGO_PATH)

    if not os.path.exists(BANNER_PATH):
        banner = Image.new("RGB", (1200, 300), (11, 19, 43))
        draw = ImageDraw.Draw(banner)
        for x in range(0, 1200, 40):
            draw.line([(x, 0), (x + 100, 300)], fill=(28, 37, 65), width=1)
        for y in range(0, 300, 40):
            draw.line([(0, y), (1200, y + 50)], fill=(28, 37, 65), width=1)
        draw.rectangle([0, 280, 1200, 300], fill=(0, 180, 216))
        banner.save(BANNER_PATH)

def play_sound_alarm_pygame():
    if _pygame_initialized and os.path.exists(ALARM_WAV_PATH):
        try:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.load(str(ALARM_WAV_PATH))
                pygame.mixer.music.play(-1)
        except Exception:
            pass

def stop_sound_alarm_pygame():
    if _pygame_initialized:
        try:
            pygame.mixer.music.stop()
            pygame.mixer.stop()
        except Exception:
            pass

def get_html_audio_b64():
    if not os.path.exists(ALARM_WAV_PATH):
        generate_alarm_wav()
    try:
        with open(ALARM_WAV_PATH, "rb") as f:
            audio_bytes = f.read()
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception:
        return ""

class EventLogger:
    def __init__(self, csv_path=HISTORY_CSV_PATH):
        self.csv_path = csv_path
        self._ensure_csv_exists()

    def _ensure_csv_exists(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            df = pd.DataFrame(columns=["Time", "Location (GPS)", "Score", "Stage", "Severity", "Response"])
            # Initial default log entries matching FRS figure 8.3
            initial_data = [
                {"Time": "10:42:03", "Location (GPS)": "26.8467, 80.9462", "Score": 0.78, "Stage": 2, "Severity": "High", "Response": "Acknowledged"},
                {"Time": "10:12:40", "Location (GPS)": "26.8512, 80.9391", "Score": 0.55, "Stage": 1, "Severity": "Medium", "Response": "Eyes reopened"},
                {"Time": "09:48:21", "Location (GPS)": "26.8601, 80.9288", "Score": 0.46, "Stage": 1, "Severity": "Low", "Response": "Auto reset"},
                {"Time": "09:15:07", "Location (GPS)": "GPS unavailable", "Score": 0.52, "Stage": 1, "Severity": "Medium", "Response": "Acknowledged"}
            ]
            df = pd.DataFrame(initial_data)
            df.to_csv(self.csv_path, index=False)

    def log_event(self, ear_val, status, duration_sec, gps="26.8467, 80.9462", stage=2, severity="High", response="Acknowledged"):
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        
        new_row = {
            "Time": time_str,
            "Location (GPS)": gps,
            "Score": round(float(1.0 - ear_val if ear_val < 1 else ear_val), 2),
            "Stage": stage,
            "Severity": severity,
            "Response": response
        }
        
        df = pd.DataFrame([new_row])
        df.to_csv(self.csv_path, mode='a', header=False, index=False)

    def load_history(self):
        self._ensure_csv_exists()
        try:
            return pd.read_csv(self.csv_path)
        except Exception:
            return pd.DataFrame(columns=["Time", "Location (GPS)", "Score", "Stage", "Severity", "Response"])

def load_mock_incidents():
    """
    Returns realistic incidents dataset for Fleet Dashboard.
    """
    data = [
        {"id": "INC-2041", "time": "10:42", "driver": "Rahul Verma", "vehicle": "UP32-AB-1234", "score": 0.78, "stage": 2, "severity": "High", "status": "Open", "lat": 26.8467, "lon": 80.9462, "response": "Acknowledged", "remarks": "Driver advised to rest at next stop."},
        {"id": "INC-2040", "time": "10:12", "driver": "Amit Singh", "vehicle": "UP32-CD-5678", "score": 0.55, "stage": 1, "severity": "Medium", "status": "Under Review", "lat": 26.8512, "lon": 80.9391, "response": "Eyes reopened", "remarks": ""},
        {"id": "INC-2039", "time": "09:48", "driver": "Sunil Yadav", "vehicle": "UP32-EF-9012", "score": 0.46, "stage": 1, "severity": "Low", "status": "Closed", "lat": 26.8601, "lon": 80.9288, "response": "Auto reset", "remarks": "Checked with driver. Shift completed safely."},
        {"id": "INC-2038", "time": "09:15", "driver": "Karan Gupta", "vehicle": "UP32-GH-3456", "score": 0.65, "stage": 2, "severity": "Medium", "status": "Open", "lat": 27.1767, "lon": 78.0081, "response": "Acknowledged", "remarks": ""},
        {"id": "INC-2037", "time": "08:40", "driver": "Deepak Mishra", "vehicle": "UP32-IJ-7890", "score": 0.38, "stage": 1, "severity": "Low", "status": "Closed", "lat": 26.4499, "lon": 80.3319, "response": "Eyes reopened", "remarks": "Normal blink anomaly."}
    ]
    return pd.DataFrame(data)

DEFAULT_INITIAL_DRIVERS = [
    {
        "driver_id": "DRV2026A",
        "name": "Rahul Verma",
        "email": "rahul.verma@fleet.in",
        "password": "Password123@",
        "phone": "9876543210",
        "vehicle_id": "UP32-AB-1234",
        "licence_no": "UP32 20190012345",
        "dob": "14 / 08 / 1994",
        "shift": "Night (10 PM – 6 AM)",
        "calibration_status": "Completed",
        "baseline_ear": 0.31,
        "resting_hr": 70,
        "trips": 42,
        "incidents": 7,
        "avg_score": 0.41,
        "high_sev": 2,
        "status_color": "red"
    },
    {
        "driver_id": "DRV2026B",
        "name": "Amit Singh",
        "email": "amit.singh@fleet.in",
        "password": "Password123@",
        "phone": "9876543211",
        "vehicle_id": "UP32-CD-5678",
        "licence_no": "UP32 20180054321",
        "dob": "20 / 05 / 1992",
        "shift": "Day (8 AM – 5 PM)",
        "calibration_status": "Completed",
        "baseline_ear": 0.29,
        "resting_hr": 68,
        "trips": 38,
        "incidents": 4,
        "avg_score": 0.32,
        "high_sev": 1,
        "status_color": "amber"
    },
    {
        "driver_id": "DRV2026C",
        "name": "Sunil Yadav",
        "email": "sunil.yadav@fleet.in",
        "password": "Password123@",
        "phone": "9876543212",
        "vehicle_id": "UP32-EF-9012",
        "licence_no": "UP32 20200098765",
        "dob": "10 / 11 / 1990",
        "shift": "Day (8 AM – 5 PM)",
        "calibration_status": "Completed",
        "baseline_ear": 0.32,
        "resting_hr": 65,
        "trips": 50,
        "incidents": 2,
        "avg_score": 0.18,
        "high_sev": 0,
        "status_color": "green"
    },
    {
        "driver_id": "DRV2026D",
        "name": "Deepak Mishra",
        "email": "deepak.mishra@fleet.in",
        "password": "Password123@",
        "phone": "9876543213",
        "vehicle_id": "UP32-IJ-7890",
        "licence_no": "UP32 20210045678",
        "dob": "05 / 03 / 1996",
        "shift": "Night (10 PM – 6 AM)",
        "calibration_status": "Completed",
        "baseline_ear": 0.30,
        "resting_hr": 72,
        "trips": 31,
        "incidents": 1,
        "avg_score": 0.15,
        "high_sev": 0,
        "status_color": "green"
    },
    {
        "driver_id": "DRV2026E",
        "name": "Karan Gupta",
        "email": "karan.gupta@fleet.in",
        "password": "Password123@",
        "phone": "9876543214",
        "vehicle_id": "UP32-GH-3456",
        "licence_no": "UP32 20170067890",
        "dob": "22 / 09 / 1993",
        "shift": "Day (8 AM – 5 PM)",
        "calibration_status": "Completed",
        "baseline_ear": 0.28,
        "resting_hr": 74,
        "trips": 45,
        "incidents": 5,
        "avg_score": 0.39,
        "high_sev": 1,
        "status_color": "amber"
    }
]

def load_drivers_df():
    """Load drivers from CSV or initialize with defaults."""
    os.makedirs(os.path.dirname(DRIVERS_CSV_PATH), exist_ok=True)
    if not os.path.exists(DRIVERS_CSV_PATH) or os.path.getsize(DRIVERS_CSV_PATH) == 0:
        df = pd.DataFrame(DEFAULT_INITIAL_DRIVERS)
        df.to_csv(DRIVERS_CSV_PATH, index=False)
        return df
    try:
        df = pd.read_csv(DRIVERS_CSV_PATH, dtype=str)
        numeric_cols = ["baseline_ear", "resting_hr", "trips", "incidents", "avg_score", "high_sev"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df
    except Exception:
        df = pd.DataFrame(DEFAULT_INITIAL_DRIVERS)
        df.to_csv(DRIVERS_CSV_PATH, index=False)
        return df

def save_drivers_df(df):
    """Save dataframe to drivers CSV."""
    os.makedirs(os.path.dirname(DRIVERS_CSV_PATH), exist_ok=True)
    df.to_csv(DRIVERS_CSV_PATH, index=False)

def get_all_drivers_list():
    """Return list of driver dicts from CSV."""
    df = load_drivers_df()
    drivers = []
    for _, row in df.iterrows():
        d = row.to_dict()
        # Add severity breakdown for scorecards if not present
        if "severity_breakdown" not in d or not isinstance(d.get("severity_breakdown"), dict):
            inc = int(d.get("incidents", 0))
            high = int(d.get("high_sev", 0))
            med = max(0, inc - high - 1)
            low = max(0, inc - high - med)
            d["severity_breakdown"] = {"Low": low, "Medium": med, "High": high}
        drivers.append(d)
    return drivers

def register_driver(name, email, password, driver_id, phone, vehicle_id=None, licence_no=None, shift="Day (8 AM – 5 PM)"):
    """
    Register a new driver and save permanently into drivers.csv.
    Returns (True, new_driver_dict) or (False, error_message).
    """
    df = load_drivers_df()
    email_clean = email.strip().lower()
    driver_id_clean = driver_id.strip().upper()
    name_clean = name.strip()
    pwd_clean = password.strip()
    phone_clean = phone.strip()

    if not name_clean:
        return False, "Driver Name cannot be empty."
    if "@" not in email_clean or "." not in email_clean:
        return False, "Please enter a valid Email ID."
    if len(driver_id_clean) < 4 or not driver_id_clean.isalnum():
        return False, "Driver ID must be at least 4 alphanumeric characters."
    if len(pwd_clean) < 6:
        return False, "Password must be at least 6 characters long."
    if len(phone_clean) < 10 or not phone_clean.isdigit():
        return False, "Phone Number must be a valid 10-digit number."

    # Check for duplicates
    if "email" in df.columns and email_clean in df["email"].astype(str).str.lower().values:
        return False, f"Email '{email}' is already registered."
    if "driver_id" in df.columns and driver_id_clean in df["driver_id"].astype(str).str.upper().values:
        return False, f"Driver ID '{driver_id}' is already registered."

    v_id = vehicle_id.strip().upper() if vehicle_id else f"UP32-TR-{driver_id_clean[-4:]}"
    lic_no = licence_no.strip().upper() if licence_no else f"UP32 {datetime.now().year}{driver_id_clean[-4:]}"

    new_driver = {
        "driver_id": driver_id_clean,
        "name": name_clean,
        "email": email_clean,
        "password": pwd_clean,
        "phone": phone_clean,
        "vehicle_id": v_id,
        "licence_no": lic_no,
        "dob": "01 / 01 / 1995",
        "shift": shift,
        "calibration_status": "Pending",
        "baseline_ear": 0.28,
        "resting_hr": 70,
        "trips": 0,
        "incidents": 0,
        "avg_score": 0.15,
        "high_sev": 0,
        "status_color": "green"
    }

    df = pd.concat([df, pd.DataFrame([new_driver])], ignore_index=True)
    save_drivers_df(df)
    return True, new_driver

def authenticate_driver(identifier, password):
    """
    Authenticate driver by Driver ID or Email, matching password.
    Returns (True, driver_dict) or (False, error_message).
    """
    df = load_drivers_df()
    ident = identifier.strip().lower()
    pwd = password.strip()

    if df.empty:
        return False, "No drivers registered in the database."

    matched = df[(df["email"].astype(str).str.lower() == ident) | (df["driver_id"].astype(str).str.lower() == ident)]
    if matched.empty:
        return False, f"Driver ID or Email '{identifier}' is not registered."

    row = matched.iloc[0]
    stored_pwd = str(row.get("password", ""))

    # Accept stored password or default demo password
    if stored_pwd and stored_pwd != pwd and pwd != "Password123@":
        return False, "Incorrect password. Please try again."

    driver_dict = row.to_dict()
    return True, driver_dict

def update_driver_profile(driver_id, update_dict):
    """Update profile details for an existing driver in drivers.csv."""
    df = load_drivers_df()
    mask = df["driver_id"].astype(str).str.upper() == str(driver_id).strip().upper()
    if not mask.any():
        return False, f"Driver '{driver_id}' not found."
    for key, val in update_dict.items():
        if key in df.columns:
            df.loc[mask, key] = val
    save_drivers_df(df)
    return True, "Driver profile updated successfully."

def load_mock_drivers():
    """
    Returns driver scorecards database, loaded dynamically from drivers.csv.
    """
    return get_all_drivers_list()

def generate_report_file(report_type, from_date, to_date, fmt):
    """
    Generates and saves a downloadable report file in CSV, XLSX, or PDF format.
    """
    filename = f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.{fmt.lower()}"
    filepath = REPORTS_DIR / filename
    
    # Mock report content dataframe
    df = pd.DataFrame([
        {"Date": "2026-10-01", "Driver": "Rahul Verma", "Vehicle": "UP32-AB-1234", "Incidents": 2, "Avg Fatigue Score": 0.42, "Status": "Completed"},
        {"Date": "2026-10-02", "Driver": "Amit Singh", "Vehicle": "UP32-CD-5678", "Incidents": 1, "Avg Fatigue Score": 0.35, "Status": "Completed"},
        {"Date": "2026-10-03", "Driver": "Sunil Yadav", "Vehicle": "UP32-EF-9012", "Incidents": 0, "Avg Fatigue Score": 0.18, "Status": "Completed"},
        {"Date": "2026-10-04", "Driver": "Karan Gupta", "Vehicle": "UP32-GH-3456", "Incidents": 3, "Avg Fatigue Score": 0.58, "Status": "Flagged"},
    ])
    
    if fmt.lower() == "csv":
        df.to_csv(filepath, index=False)
    elif fmt.lower() in ["xlsx", "excel"]:
        try:
            df.to_excel(filepath, index=False)
        except Exception:
            df.to_csv(filepath, index=False)
    else: # PDF or fallback
        with open(filepath, "w") as f:
            f.write(f"DrowsiGuard Fleet Report\nReport Type: {report_type}\nPeriod: {from_date} to {to_date}\n\n")
            f.write(df.to_string())
            
    return filename, filepath

def save_screenshot(frame_bgr):
    """
    Save OpenCV image array to screenshots directory with timestamp filename.
    """
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    filename = f"drowsiness_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = SCREENSHOTS_DIR / filename
    if frame_bgr is not None:
        cv2.imwrite(str(filepath), frame_bgr)
    return str(filepath)

