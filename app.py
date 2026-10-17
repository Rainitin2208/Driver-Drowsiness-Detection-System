"""
DrowsiGuard - AI-Based Driver Drowsiness Detection and Alert System
Main Application providing both:
1. 📱 In-Vehicle Edge Unit Display (Dark Theme Driver Panel)
2. 🌐 Fleet & Cloud Dashboard (Fleet Safety Platform Web Dashboard)

Full implementation of the Functional Requirement Specification (FRS).
"""

import time
import os
import datetime
import cv2
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from config import (
    DEFAULT_EAR_THRESHOLD, DEFAULT_CONSEC_FRAMES, DEFAULT_DRIVER,
    THRESHOLD_RANGES, LOGO_PATH, BANNER_PATH, HISTORY_CSV_PATH, COLORS
)
from utils import (
    generate_alarm_wav, generate_branding_assets,
    EventLogger, save_screenshot, get_html_audio_b64,
    stop_sound_alarm_pygame, load_mock_incidents, load_mock_drivers,
    generate_report_file, register_driver, authenticate_driver,
    update_driver_profile, load_drivers_df
)
from detector import DrowsinessDetector

# Page Configuration
st.set_page_config(
    page_title="DrowsiGuard - AI Driver Drowsiness & Fleet Safety Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure assets
generate_alarm_wav()
generate_branding_assets()

# Initialize Detector in session state
if "detector" not in st.session_state:
    st.session_state.detector = DrowsinessDetector()
if "drivers_list" not in st.session_state:
    st.session_state.drivers_list = load_mock_drivers()
if "current_driver" not in st.session_state:
    st.session_state.current_driver = st.session_state.drivers_list[0] if st.session_state.drivers_list else DEFAULT_DRIVER.copy()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = True
if "incidents_df" not in st.session_state:
    st.session_state.incidents_df = load_mock_incidents()
if "thresholds" not in st.session_state:
    st.session_state.thresholds = {
        "eye_closure": 2.0,
        "yawn_count": 3,
        "head_pose": 25,
        "warning_score": 0.50,
        "critical_score": 0.75,
        "escalation_window": 10
    }

# Inject Custom CSS for Pixel-Perfect DrowsiGuard Panels
def inject_custom_css(mode="Edge"):
    if mode == "Edge":
        st.markdown("""
        <style>
        /* Dark Theme In-Vehicle Edge Unit Styling */
        .stApp {
            background-color: #0B132B !important;
            color: #E2E8F0 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Top Status Bar */
        .edge-top-bar {
            background-color: #1C2541;
            border-bottom: 2px solid #00B4D8;
            padding: 12px 24px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .edge-brand {
            font-size: 1.5rem;
            font-weight: 800;
            color: #00B4D8;
            letter-spacing: 0.5px;
        }
        .edge-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #FFFFFF;
        }
        .edge-status {
            font-size: 0.95rem;
            font-weight: 600;
            color: #94A3B8;
            letter-spacing: 2px;
        }
        
        /* Bottom Footer Bar */
        .edge-bottom-bar {
            background-color: #1C2541;
            border-top: 1px solid #334155;
            padding: 10px 24px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 30px;
            color: #94A3B8;
            font-size: 0.9rem;
        }
        
        /* Metric Card Container */
        .edge-card {
            background: #1C2541;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .edge-metric-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            color: #94A3B8;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }
        .edge-metric-val {
            font-size: 2rem;
            font-weight: 800;
            color: #FFFFFF;
        }
        
        /* Badges */
        .status-alert {
            background-color: rgba(16, 185, 129, 0.2);
            color: #10B981;
            border: 2px solid #10B981;
            font-size: 2.2rem;
            font-weight: 900;
            padding: 12px 24px;
            border-radius: 10px;
            text-align: center;
            letter-spacing: 1px;
        }
        .status-warning {
            background-color: rgba(245, 158, 11, 0.2);
            color: #F59E0B;
            border: 2px solid #F59E0B;
            font-size: 2.2rem;
            font-weight: 900;
            padding: 12px 24px;
            border-radius: 10px;
            text-align: center;
            letter-spacing: 1px;
        }
        .status-drowsy {
            background-color: rgba(239, 68, 68, 0.2);
            color: #EF4444;
            border: 2px solid #EF4444;
            font-size: 2.2rem;
            font-weight: 900;
            padding: 12px 24px;
            border-radius: 10px;
            text-align: center;
            letter-spacing: 1px;
            animation: pulse 1.5s infinite;
        }
        
        /* High Alert Full Screen Overlay */
        .drowsy-alert-overlay {
            background-color: #4A0E0E;
            border: 3px solid #EF4444;
            padding: 40px;
            border-radius: 16px;
            text-align: center;
            color: white;
            box-shadow: 0 0 40px rgba(239, 68, 68, 0.6);
        }
        .drowsy-alert-title {
            font-size: 3rem;
            font-weight: 900;
            color: #FFFFFF;
            letter-spacing: 2px;
            margin-top: 15px;
        }
        
        /* Tile Grid for Settings */
        .settings-tile {
            background: #1C2541;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 30px 15px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 700;
            color: #F8FAFC;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .settings-tile:hover {
            border-color: #00B4D8;
            background: #273459;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        /* Light Theme Fleet Dashboard Styling */
        .stApp {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Mock Browser Header */
        .browser-header {
            background-color: #E2E8F0;
            border-radius: 10px 10px 0 0;
            padding: 8px 16px;
            display: flex;
            align-items: center;
            border-bottom: 1px solid #CBD5E1;
            margin-bottom: 20px;
        }
        .browser-dots {
            display: flex;
            gap: 6px;
            margin-right: 15px;
        }
        .dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        .dot-red { background: #FF5F56; }
        .dot-yellow { background: #FFBD2E; }
        .dot-green { background: #27C93F; }
        .browser-url-bar {
            background: #FFFFFF;
            padding: 4px 16px;
            border-radius: 16px;
            font-family: monospace;
            font-size: 0.85rem;
            color: #475569;
            flex-grow: 1;
            border: 1px solid #CBD5E1;
        }
        
        /* KPI Card */
        .fleet-kpi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 18px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        }
        .kpi-title {
            font-size: 0.85rem;
            color: #64748B;
            font-weight: 600;
        }
        .kpi-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: #0F172A;
            margin-top: 4px;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)


# Render Top Header Component for In-Vehicle Unit
def render_edge_header(screen_name):
    st.markdown(f"""
    <div class="edge-top-bar">
        <div class="edge-brand">DrowsiGuard</div>
        <div class="edge-title">{screen_name}</div>
        <div class="edge-status">GPS &nbsp; CAM &nbsp; BT &nbsp; 10:42</div>
    </div>
    """, unsafe_allow_html=True)

# Render Bottom Footer Component for In-Vehicle Unit
def render_edge_footer():
    driver = st.session_state.current_driver
    st.markdown(f"""
    <div class="edge-bottom-bar">
        <div>Driver: {driver['name']} &nbsp;|&nbsp; Vehicle: {driver['vehicle_id']}</div>
        <div>Edge Unit v1.0</div>
    </div>
    """, unsafe_allow_html=True)

# Render Mock Browser Header for Fleet Dashboard
def render_browser_header(url_path):
    st.markdown(f"""
    <div class="browser-header">
        <div class="browser-dots">
            <div class="dot dot-red"></div>
            <div class="dot dot-yellow"></div>
            <div class="dot dot-green"></div>
        </div>
        <div class="browser-url-bar">https://fleet.drowsiguard.in/{url_path}</div>
        <div style="font-weight: 600; font-size: 0.9rem; color: #334155; margin-left: 15px;">Fleet Manager ▼</div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# MAIN APP ENTRY POINT
# ==============================================================================
def main():
    # Sidebar Global View Mode Selector
    st.sidebar.image(str(LOGO_PATH), width=130)
    st.sidebar.markdown("### 🛡️ **DrowsiGuard System**")
    
    view_mode = st.sidebar.radio(
        "Select System Panel",
        ["📱 In-Vehicle Unit (Driver Display)", "🌐 Fleet & Cloud Dashboard (Web Platform)"],
        index=0
    )
    
    detector = st.session_state.detector
    
    if view_mode == "📱 In-Vehicle Unit (Driver Display)":
        inject_custom_css("Edge")
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧭 **In-Vehicle Navigation**")
        
        edge_screen = st.sidebar.selectbox(
            "Select Screen Module",
            [
                "6.2 Live Monitoring Screen",
                "6.1 Camera & Driver Calibration",
                "3. Splash Screen",
                "4.1 Driver Login",
                "4.2 Forgot Password",
                "6.3 Basic Drowsiness Alert",
                "7.1 Wearable Pairing",
                "7.2 Yawning & Head-Pose Monitoring",
                "7.3 Fused Fatigue Score",
                "8.1 Escalating Alerts",
                "8.2 Alert Configuration",
                "8.3 Geo-Tagged Incident Log",
                "11.1 Settings Grid",
                "11.2 Change Password",
                "11.3 Notifications Preferences",
                "11.4 Report Issue / Help",
                "11.5 Terms of Use",
                "11.6 Logout Confirmation"
            ],
            index=0
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### ⚙️ Quick Controls")
        ear_thresh = st.sidebar.slider("EAR Threshold (Eye)", 0.15, 0.35, detector.ear_threshold, 0.01)
        consec_fr = st.sidebar.slider("Consecutive Frames", 5, 50, detector.consec_frames, 1)
        yawn_thresh = st.sidebar.slider("Yawn MAR Threshold (Mouth)", 0.25, 0.70, detector.yawn_mar_threshold, 0.02)
        detector.update_settings(ear_thresh, consec_fr, detector.alarm_enabled)
        detector.yawn_mar_threshold = yawn_thresh

        
        # Dispatch Screen
        if edge_screen == "3. Splash Screen":
            render_splash_screen()
        elif edge_screen == "4.1 Driver Login":
            render_driver_login()
        elif edge_screen == "4.2 Forgot Password":
            render_forgot_password()
        elif edge_screen == "6.1 Camera & Driver Calibration":
            render_calibration_screen()
        elif edge_screen == "6.2 Live Monitoring Screen":
            render_live_monitoring()
        elif edge_screen == "6.3 Basic Drowsiness Alert":
            render_basic_alert()
        elif edge_screen == "7.1 Wearable Pairing":
            render_wearable_pairing()
        elif edge_screen == "7.2 Yawning & Head-Pose Monitoring":
            render_yawn_headpose_monitoring()
        elif edge_screen == "7.3 Fused Fatigue Score":
            render_fused_fatigue_score()
        elif edge_screen == "8.1 Escalating Alerts":
            render_escalating_alerts()
        elif edge_screen == "8.2 Alert Configuration":
            render_alert_configuration()
        elif edge_screen == "8.3 Geo-Tagged Incident Log":
            render_geotagged_incident_log()
        elif edge_screen == "11.1 Settings Grid":
            render_settings_grid()
        elif edge_screen == "11.2 Change Password":
            render_change_password()
        elif edge_screen == "11.3 Notifications Preferences":
            render_notifications_preferences()
        elif edge_screen == "11.4 Report Issue / Help":
            render_report_issue()
        elif edge_screen == "11.5 Terms of Use":
            render_terms_of_use()
        elif edge_screen == "11.6 Logout Confirmation":
            render_logout_confirmation()
            
    else: # Fleet Dashboard
        inject_custom_css("Fleet")
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🏢 **Fleet Navigation**")
        
        fleet_screen = st.sidebar.radio(
            "Fleet Portal",
            [
                "Dashboard (Home)",
                "Create Driver Account",
                "Driver Profile & Vehicle Assignment",
                "Driver Scorecards",
                "Incident Review Workflow",
                "Threshold Configuration",
                "Fatigue-Risk Prediction",
                "Route-Level Risk Analytics",
                "Reports & Data Export",
                "Fleet Manager Login"
            ]
        )
        
        if fleet_screen == "Dashboard (Home)":
            render_fleet_home()
        elif fleet_screen == "Create Driver Account":
            render_create_driver()
        elif fleet_screen == "Driver Profile & Vehicle Assignment":
            render_driver_profile_assignment()
        elif fleet_screen == "Driver Scorecards":
            render_driver_scorecards()
        elif fleet_screen == "Incident Review Workflow":
            render_incident_review()
        elif fleet_screen == "Threshold Configuration":
            render_threshold_configuration()
        elif fleet_screen == "Fatigue-Risk Prediction":
            render_fatigue_risk_prediction()
        elif fleet_screen == "Route-Level Risk Analytics":
            render_route_analytics()
        elif fleet_screen == "Reports & Data Export":
            render_reports_export()
        elif fleet_screen == "Fleet Manager Login":
            render_fleet_login()


# ==============================================================================
# IN-VEHICLE EDGE UNIT SCREENS IMPLEMENTATION
# ==============================================================================

def render_splash_screen():
    render_edge_header("Splash Screen")
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style="text-align: center; background: #1C2541; padding: 40px; border-radius: 16px; border: 1px solid #334155;">
                <div style="font-size: 5rem; color: #00B4D8; margin-bottom: 10px;">👁️</div>
                <h1 style="color: #FFFFFF; font-weight: 800; margin: 0;">DrowsiGuard</h1>
                <p style="color: #94A3B8; font-size: 1.1rem; margin-top: 5px;">AI-Based Driver Drowsiness Detection and Alert System</p>
                <br>
                <div style="display: flex; justify-content: space-around; font-size: 1rem; color: #10B981; font-weight: 700; margin: 20px 0;">
                    <span>✓ Camera</span>
                    <span>✓ GPS</span>
                    <span>✓ Buzzer</span>
                    <span style="color: #F59E0B;">... Vibration Motor</span>
                </div>
                <div style="background-color: #0B132B; border-radius: 10px; height: 12px; overflow: hidden; margin: 20px 0;">
                    <div style="background-color: #00B4D8; width: 65%; height: 100%; border-radius: 10px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.checkbox("Simulate Hardware Validation Error (Camera Not Detected)"):
            st.error("❌ Validation Message: Camera Not Detected. Please Check The Connection")
            
    render_edge_footer()

def render_driver_login():
    render_edge_header("Driver Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        current_drv = st.session_state.get("current_driver", {})
        is_logged = st.session_state.get("logged_in", False)
        
        st.markdown(
            f"""
            <div style="background: #1C2541; padding: 25px; border-radius: 16px; border: 1px solid #334155; text-align: center; margin-bottom: 20px;">
                <h2 style="color: #FFFFFF; margin: 0;">DrowsiGuard Driver Portal</h2>
                <p style="color: #94A3B8; margin-top: 5px;">Active: <strong style="color: #00B4D8;">{current_drv.get('name', 'None')}</strong> ({current_drv.get('driver_id', 'N/A')})</p>
                <div style="font-size: 0.85rem; color: #10B981; font-weight: 600;">Status: {'🟢 Active Session' if is_logged else '⚪ Not Logged In'}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        login_tab, register_tab = st.tabs(["🔑 Driver Login", "➕ Register New Driver"])
        
        with login_tab:
            with st.form("driver_login_form"):
                driver_id = st.text_input("Driver ID or Email", value=current_drv.get("email", "rahul.verma@fleet.in"), max_chars=40)
                password = st.text_input("Password", value="Password123@", type="password", max_chars=30)
                
                c_forgot, c_sub = st.columns([1, 1])
                with c_forgot:
                    st.caption("[Forgot Password?](#)")
                with c_sub:
                    submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
                    
                if submitted:
                    if not driver_id:
                        st.error("Please enter your Driver ID or Email.")
                    elif not password:
                        st.error("Please enter your Password.")
                    else:
                        success, result = authenticate_driver(driver_id, password)
                        if success:
                            st.session_state.current_driver = result
                            st.session_state.logged_in = True
                            st.session_state.drivers_list = load_mock_drivers()
                            st.success(f"🎉 Login Successful! Welcome back, {result['name']}.")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"❌ {result}")
            
            with st.expander("ℹ️ Demo Credentials (Click to View)"):
                st.markdown("""
                You can log in with any registered fleet driver (Password: `Password123@`):
                - **Rahul Verma**: `rahul.verma@fleet.in` or `DRV2026A`
                - **Amit Singh**: `amit.singh@fleet.in` or `DRV2026B`
                - **Sunil Yadav**: `sunil.yadav@fleet.in` or `DRV2026C`
                - **Deepak Mishra**: `deepak.mishra@fleet.in` or `DRV2026D`
                - **Karan Gupta**: `karan.gupta@fleet.in` or `DRV2026E`
                - *Or create your own driver in the 'Register New Driver' tab!*
                """)
                
        with register_tab:
            with st.form("driver_quick_register_form"):
                reg_name = st.text_input("Driver Full Name *", placeholder="e.g. Nitin Rai")
                c_r1, c_r2 = st.columns(2)
                with c_r1:
                    reg_email = st.text_input("Email ID *", placeholder="e.g. nitin@fleet.in")
                    reg_id = st.text_input("Driver ID *", placeholder="e.g. DRV2026N")
                with c_r2:
                    reg_phone = st.text_input("Phone Number * (10 digits)", placeholder="e.g. 9876543210")
                    reg_vehicle = st.text_input("Vehicle Number", placeholder="e.g. UP32-NR-1001")
                    
                c_p1, c_p2 = st.columns(2)
                with c_p1:
                    reg_password = st.text_input("Create Password *", type="password")
                with c_p2:
                    reg_confirm = st.text_input("Confirm Password *", type="password")
                    
                reg_submit = st.form_submit_button("Register & Login", use_container_width=True, type="primary")
                if reg_submit:
                    if reg_password != reg_confirm:
                        st.error("Passwords do not match. Please re-enter.")
                    else:
                        success, res = register_driver(
                            name=reg_name,
                            email=reg_email,
                            password=reg_password,
                            driver_id=reg_id,
                            phone=reg_phone,
                            vehicle_id=reg_vehicle
                        )
                        if success:
                            st.session_state.current_driver = res
                            st.session_state.logged_in = True
                            st.session_state.drivers_list = load_mock_drivers()
                            st.success(f"✅ Driver '{reg_name}' registered successfully and logged in!")
                            time.sleep(0.6)
                            st.rerun()
                        else:
                            st.error(f"❌ {res}")
                            
    render_edge_footer()

def render_forgot_password():
    render_edge_header("Forgot Password")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style="background: #1C2541; padding: 30px; border-radius: 16px; border: 1px solid #334155; text-align: center;">
                <h2 style="color: #FFFFFF; margin: 0;">Reset your password</h2>
                <p style="color: #94A3B8;">Enter your registered email. A reset link will be sent.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        email = st.text_input("Registered Email ID", value="rahul.verma@fleet.in", max_chars=30)
        
        col_send, col_back = st.columns([1, 1])
        with col_send:
            if st.button("Send", use_container_width=True, type="primary"):
                if "@" not in email or "." not in email:
                    st.error("Please Enter a Valid Email ID")
                else:
                    st.success("Reset link sent to registered email!")
        with col_back:
            st.button("← Back to Login", use_container_width=True)
            
    render_edge_footer()

def render_calibration_screen():
    render_edge_header("Camera & Driver Calibration")
    detector = st.session_state.detector
    
    col_cam, col_calib = st.columns([3, 2])
    
    with col_cam:
        cam_calib_ph = st.empty()
        cam_calib_ph.markdown(
            """
            <div style="position: relative; background: #000; border-radius: 12px; overflow: hidden; border: 2px solid #10B981; height: 380px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 20px;">
                <div style="font-size: 3.5rem; margin-bottom: 10px;">👤</div>
                <h4 style="color: #FFFFFF; margin: 0;">Camera Calibration Standby</h4>
                <p style="color: #94A3B8; font-size: 0.85rem; max-width: 320px; margin-top: 6px;">
                    Look directly at the webcam with open eyes, then click "Start Live Calibration" to measure and record your personalized baseline Eye Aspect Ratio (EAR).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_calib:
        calib_info_ph = st.empty()
        baseline_display_val = st.session_state.current_driver.get("baseline_ear", 0.31)
        
        calib_info_ph.markdown(
            f"""
            <div class="edge-card">
                <h3 style="color: #FFFFFF; margin-top: 0;">Calibration Setup</h3>
                <div style="color: #10B981; font-weight: 700; margin-bottom: 8px;">🟢 Camera: Ready (OpenCV)</div>
                <div style="color: #10B981; font-weight: 700; margin-bottom: 8px;">🟢 Status: {st.session_state.current_driver.get('calibration_status', 'Pending')}</div>
                <div style="color: #F59E0B; font-weight: 700; margin-bottom: 16px;">🟠 Baseline EAR: {baseline_display_val:.2f}</div>
                
                <div style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase;">Active Baseline EAR</div>
                <div style="font-size: 2.8rem; font-weight: 900; color: #00B4D8;">{baseline_display_val:.2f}</div>
                <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">Drowsiness Threshold (75%): {round(baseline_display_val * 0.75, 2):.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if st.button("Start Live Calibration", type="primary", use_container_width=True):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
                
            if cap.isOpened():
                progress_bar = st.progress(0, text="Calibrating eyes... please look at camera")
                collected_ears = []
                frame_idx = 0
                total_frames = 25
                
                while frame_idx < total_frames and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    annotated = detector.process_frame(frame)
                    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    cam_calib_ph.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    if detector.face_detected and detector.current_ear > 0.05:
                        collected_ears.append(detector.current_ear)
                        frame_idx += 1
                        progress_bar.progress(int((frame_idx / total_frames) * 100), text=f"Calibrating... {frame_idx}/{total_frames} frames")
                    time.sleep(0.04)
                    
                cap.release()
                if collected_ears:
                    avg_ear = round(float(np.mean(collected_ears)), 3)
                    calc_thresh = round(avg_ear * 0.75, 2)
                    detector.ear_threshold = calc_thresh
                    st.session_state.current_driver["baseline_ear"] = avg_ear
                    st.session_state.current_driver["calibration_status"] = "Completed"
                    st.success(f"Calibration Completed! Baseline EAR: {avg_ear:.3f}. Threshold set to {calc_thresh:.2f}")
                    st.rerun()
                else:
                    st.warning("Could not detect face clearly during calibration. Please ensure good lighting and look directly at camera.")
            else:
                st.error("Hardware Camera (OpenCV Device 0) not accessible. Please check your camera connection or close other apps using it.")
                
        if st.button("Skip (use last calibration)", use_container_width=True):
            st.info("Using saved default calibration.")
            
    render_edge_footer()

def render_live_monitoring():
    render_edge_header("Live Monitoring")
    detector = st.session_state.detector
    
    col_left, col_right = st.columns([3, 3])
    
    with col_left:
        st.markdown("#### 📹 Live Video Stream & Vision Engine")
        
        # Camera Mode Selector
        cam_source = st.radio(
            "Camera Feed Source", 
            ["📹 Direct OpenCV Webcam (Fast & Real-time)", "🌐 WebRTC Browser Camera"], 
            horizontal=True
        )
        
        if cam_source == "🌐 WebRTC Browser Camera":
            from detector import DrowsinessVideoProcessor
            DrowsinessVideoProcessor.detector_ref = detector
            
            rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            webrtc_ctx = webrtc_streamer(
                key="drowsiguard-live-monitoring-v2",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_config,
                video_processor_factory=DrowsinessVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            st.caption("ℹ️ Click 'START' in the box above to stream directly from your browser.")
            frame_placeholder = None
            cam_device_idx = 0
            is_active_cam = False
        else: # Direct OpenCV Webcam
            c_dev, c_tgl = st.columns([1, 2])
            with c_dev:
                cam_device_idx = st.selectbox("Camera Device", [0, 1, 2], index=0, format_func=lambda x: f"Device {x}")
            with c_tgl:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                is_active_cam = st.toggle("🟢 Turn ON Live OpenCV Stream", value=st.session_state.get("cam_toggle_state", False), key="cam_toggle_state")
                
            frame_placeholder = st.empty()
            
        fatigue_placeholder = st.empty()

    with col_right:
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        # Action Buttons
        b1, b2, b3 = st.columns([1, 1, 1])
        with b1:
            if st.button("Start Trip", type="primary", use_container_width=True):
                detector.trip_active = True
                detector.session_start_time = time.time()
                st.success("Trip Started!")
        with b2:
            if st.button("End Trip", use_container_width=True):
                detector.trip_active = False
                st.info("Trip Concluded.")
        with b3:
            mute_label = "Unmute Chime" if detector.muted_chime else "Mute Chime"
            if st.button(mute_label, use_container_width=True):
                detector.muted_chime = not detector.muted_chime
                st.rerun()
            
        recent_events_placeholder = st.empty()

    # Function to update all metric cards simultaneously
    def _update_ui_cards():
        status_class = "status-alert" if not detector.is_drowsy else "status-drowsy"
        status_text = "ALERT" if not detector.is_drowsy else "DROWSY"
        status_tag_bg = "rgba(16, 185, 129, 0.2)" if not detector.is_drowsy else "rgba(239, 68, 68, 0.2)"
        status_tag_color = "#10B981" if not detector.is_drowsy else "#EF4444"
        status_tag_border = "#10B981" if not detector.is_drowsy else "#EF4444"
        status_tag_text = "Attentive & Active" if not detector.is_drowsy else "DROWSINESS DETECTED!"
        
        elapsed_trip_s = int(time.time() - detector.session_start_time)
        hrs = elapsed_trip_s // 3600
        mins = (elapsed_trip_s % 3600) // 60
        secs = elapsed_trip_s % 60
        trip_time_str = f"Trip {hrs:02d}:{mins:02d}:{secs:02d}"

        status_placeholder.markdown(
            f"""
            <div class="edge-card" style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="color: #94A3B8; font-size: 0.8rem; text-transform: uppercase;">Driver Status</div>
                    <div class="{status_class}" style="display: inline-block; margin-top: 4px;">{status_text}</div>
                </div>
                <div style="text-align: right;">
                    <div style="background: #0B132B; padding: 6px 12px; border-radius: 20px; font-weight: 700; color: #E2E8F0; font-size: 0.9rem;">{trip_time_str}</div>
                    <div style="background: {status_tag_bg}; color: {status_tag_color}; border: 1px solid {status_tag_border}; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; margin-top: 8px; display: inline-block;">{status_tag_text}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        ear_color = "#EF4444" if detector.current_ear < detector.ear_threshold else "#10B981"
        mar_color = "#F59E0B" if detector.is_yawning else "#00B4D8"
        face_status = "Detected ✓" if detector.face_detected else "Searching..."

        metrics_placeholder.markdown(
            f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin: 12px 0;">
                <div class="edge-card">
                    <div class="edge-metric-label">EAR (Eye Ratio)</div>
                    <div class="edge-metric-val" style="color: {ear_color};">{detector.current_ear:.3f}</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">Thresh: {detector.ear_threshold:.2f} ({'CLOSED' if detector.current_ear < detector.ear_threshold else 'OPEN'})</div>
                </div>
                <div class="edge-card">
                    <div class="edge-metric-label">Blink rate</div>
                    <div class="edge-metric-val">{detector.blink_count}<span style="font-size:0.85rem; font-weight:400;"> blinks</span></div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">Face: {face_status}</div>
                </div>
                <div class="edge-card">
                    <div class="edge-metric-label">Alert Stage</div>
                    <div class="edge-metric-val" style="color: {'#EF4444' if detector.alert_stage > 0 else '#10B981'};">Stage {detector.alert_stage}</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">Alerts: {detector.total_alerts}</div>
                </div>
                <div class="edge-card">
                    <div class="edge-metric-label">Yawns (MAR)</div>
                    <div class="edge-metric-val" style="color: {mar_color};">{detector.yawn_count}</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">MAR: {detector.current_mar:.2f} (Limit {detector.yawn_mar_threshold:.2f})</div>
                </div>
                <div class="edge-card">
                    <div class="edge-metric-label">Head Pose Tilt</div>
                    <div class="edge-metric-val">{abs(int(detector.pitch))}°</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">Pitch {detector.pitch:+.0f}° | Yaw {detector.yaw:+.0f}°</div>
                </div>
                <div class="edge-card">
                    <div class="edge-metric-label">Heart rate</div>
                    <div class="edge-metric-val">{detector.heart_rate}<span style="font-size:0.85rem; font-weight:400;"> bpm</span></div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">HRV: {detector.hrv} ms</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        score_pct = int(detector.fused_score * 100)
        score_color = "#10B981" if detector.fused_score < 0.4 else ("#F59E0B" if detector.fused_score < 0.7 else "#EF4444")
        risk_text = "Low risk" if detector.fused_score < 0.4 else ("Moderate risk" if detector.fused_score < 0.7 else "CRITICAL DROWSINESS RISK")

        fatigue_placeholder.markdown(
            f"""
            <div class="edge-card">
                <div style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase;">Fatigue Score (fused)</div>
                <div style="background-color: #0B132B; border-radius: 8px; height: 14px; margin: 8px 0;">
                    <div style="background-color: {score_color}; width: {min(100, score_pct)}%; height: 100%; border-radius: 8px; transition: width 0.3s ease;"></div>
                </div>
                <div style="font-size: 1.1rem; font-weight: 800; color: {score_color};">
                    {detector.fused_score} • {risk_text}
                </div>
                <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">
                    Signals: Eye {'✓' if detector.current_ear >= detector.ear_threshold else '⚠'} &nbsp; Yawn {'✓' if not detector.is_yawning else '⚠'} &nbsp; Head ✓ &nbsp; HR ✓ &nbsp; Steering ✓
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        recent_events_placeholder.markdown(
            f"""
            <div class="edge-card" style="margin-top: 10px;">
                <div style="color: #94A3B8; font-size: 0.85rem; font-weight: 700; margin-bottom: 8px;">Recent events</div>
                <div style="font-size: 0.85rem; color: #CBD5E1; margin-bottom: 4px;">{'⚠️ Drowsiness event detected!' if detector.is_drowsy else ('⚠️ Yawn detected!' if detector.is_yawning else '✓ Driver attentive — No active events')}</div>
                <div style="font-size: 0.85rem; color: #94A3B8;">Total drowsiness alerts this trip: {detector.total_alerts}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Initial static render of cards
    _update_ui_cards()

    # OpenCV Live Camera Stream Execution
    if cam_source != "🌐 WebRTC Browser Camera" and frame_placeholder is not None:
        if is_active_cam:
            cap = cv2.VideoCapture(cam_device_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(cam_device_idx)
            
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                st.toast(f"OpenCV Camera (Device {cam_device_idx}) connected!", icon="📹")
                
                try:
                    while st.session_state.get("cam_toggle_state", False):
                        ret, frame = cap.read()
                        if not ret:
                            frame_placeholder.warning("Waiting for webcam frame...")
                            time.sleep(0.05)
                            continue
                        
                        annotated = detector.process_frame(frame)
                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                        _update_ui_cards()
                        time.sleep(0.01)
                finally:
                    cap.release()
            else:
                frame_placeholder.error(
                    f"❌ Hardware Camera (OpenCV Device Index {cam_device_idx}) could not be opened.\n\n"
                    "💡 Tips to resolve:\n"
                    "1. Try selecting a different Camera Device (e.g. Device 1 or 2) from the dropdown above.\n"
                    "2. Make sure no other application (Zoom, Teams, Camera app, or another browser tab) is using your webcam.\n"
                    "3. Check Windows Camera Privacy Settings (Settings > Privacy & security > Camera > 'Let desktop apps access your camera')."
                )
        else:
            frame_placeholder.markdown(
                """
                <div style="background: #0B132B; border: 2px dashed #334155; border-radius: 12px; height: 350px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 20px;">
                    <div style="font-size: 3.5rem; margin-bottom: 10px;">📹</div>
                    <h4 style="color: #FFFFFF; margin: 0 0 8px 0;">OpenCV Camera is in Standby</h4>
                    <p style="color: #94A3B8; font-size: 0.9rem; max-width: 400px; margin: 0;">
                        Switch the toggle <strong>'🟢 Turn ON Live OpenCV Stream'</strong> above to begin real-time eye tracking, yawn detection, and drowsiness monitoring.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    render_edge_footer()

def render_basic_alert():
    render_edge_header("Basic Drowsiness Alert")
    detector = st.session_state.detector
    
    st.markdown(
        """
        <div class="drowsy-alert-overlay">
            <div style="font-size: 4rem;">⚠️</div>
            <div class="drowsy-alert-title">DROWSINESS DETECTED</div>
            <div style="font-size: 1.5rem; color: #FCA5A5; margin-top: 10px;">Please take a break</div>
            <div style="font-size: 1.1rem; color: #F8FAFC; margin: 20px 0;">Eyes closed for 2.4 s (threshold 2.0 s)</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("ACKNOWLEDGE", type="primary", use_container_width=True):
            detector.acknowledge_alert()
            st.success("Alert Acknowledged. Status Reset to Normal.")
            
        st.markdown(
            """
            <div class="edge-card" style="margin-top: 20px; border-left: 4px solid #EF4444;">
                <div style="font-weight: 800; color: #EF4444;">LED: ON &nbsp;&nbsp;&nbsp; Buzzer: ON</div>
                <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">Event logged 10:42:03</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    render_edge_footer()

def render_wearable_pairing():
    render_edge_header("Wearable Pairing")
    detector = st.session_state.detector
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown(
            """
            <div class="edge-card">
                <h3 style="color: #FFFFFF; margin-top: 0;">Nearby Bluetooth devices</h3>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #334155;">
                    <div>
                        <div style="font-weight: 700; color: #F8FAFC;">Mi Smart Band 7</div>
                        <div style="font-size: 0.8rem; color: #94A3B8;">Signal: strong</div>
                    </div>
                    <span style="color: #00B4D8; font-weight: 700; cursor: pointer;">Pair</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #334155;">
                    <div>
                        <div style="font-weight: 700; color: #F8FAFC;">Noise ColorFit Pro</div>
                        <div style="font-size: 0.8rem; color: #94A3B8;">Signal: medium</div>
                    </div>
                    <span style="color: #00B4D8; font-weight: 700; cursor: pointer;">Pair</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 700; color: #F8FAFC;">Amazfit Band 5</div>
                        <div style="font-size: 0.8rem; color: #94A3B8;">Signal: weak</div>
                    </div>
                    <span style="color: #00B4D8; font-weight: 700; cursor: pointer;">Pair</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.button("Scan Again", use_container_width=True)

    with c2:
        st.markdown(
            """
            <div class="edge-card" style="text-align: center;">
                <div style="color: #FFFFFF; font-size: 1.1rem; font-weight: 700;">Connected device</div>
                <div style="color: #10B981; font-weight: 700; font-size: 0.9rem; margin-top: 4px;">Mi Smart Band 7 • Connected</div>
                
                <div style="font-size: 4rem; font-weight: 900; color: #EF4444; margin: 15px 0;">72 <span style="font-size: 1.2rem; color: #94A3B8;">bpm</span></div>
                <div style="color: #94A3B8; font-size: 0.9rem;">HRV: 48 ms</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Live Heart Rate Waveform Sparkline
        x_vals = np.linspace(0, 10, 50)
        y_vals = 72 + 3 * np.sin(x_vals * 2) + np.random.normal(0, 0.5, 50)
        fig = px.line(x=x_vals, y=y_vals, labels={'x': 'Time', 'y': 'BPM'})
        fig.update_traces(line_color='#EF4444', line_width=3)
        fig.update_layout(height=120, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_visible=False, yaxis_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.button("Unpair", type="primary", use_container_width=True)

    render_edge_footer()

def render_yawn_headpose_monitoring():
    render_edge_header("Yawning & Head-Pose Monitoring")
    detector = st.session_state.detector
    
    col_cam, col_metrics = st.columns([3, 2])
    with col_cam:
        st.markdown(
            f"""
            <div style="position: relative; background: #000; border-radius: 12px; overflow: hidden; border: 2px solid #00B4D8; height: 350px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <div style="position: absolute; top: 15px; left: 15px; color: {'#10B981' if detector.face_detected else '#EF4444'}; font-weight: 700;">● {'TRACKING ACTIVE' if detector.face_detected else 'SEARCHING FACE'}</div>
                <div style="position: absolute; top: 15px; right: 15px; color: #10B981; font-weight: 700;">Face Mesh</div>
                <div style="text-align: center;">
                    <div style="font-size: 4rem;">👤</div>
                    <div style="color: {'#F59E0B' if detector.is_yawning else '#10B981'}; font-weight: 700; margin-top: 10px; font-size: 1.3rem;">
                        MAR {detector.current_mar:.2f} ({'YAWN DETECTED!' if detector.is_yawning else 'Normal'})
                    </div>
                    <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">Limit Threshold: {detector.yawn_mar_threshold:.2f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_metrics:
        c1, c2 = st.columns(2)
        with c1:
            yawn_status = "WARNING" if detector.yawn_count >= 3 else ("NORMAL" if detector.yawn_count == 0 else "CAUTION")
            yawn_color = "#EF4444" if detector.yawn_count >= 3 else ("#10B981" if detector.yawn_count == 0 else "#F59E0B")
            st.markdown(
                f"""
                <div class="edge-card">
                    <div style="color: #94A3B8; font-size: 0.8rem;">Yawn count (trip)</div>
                    <div style="font-size: 3rem; font-weight: 900; color: {yawn_color};">{detector.yawn_count}</div>
                    <div style="font-size: 0.75rem; color: #94A3B8;">Limit: 3 in 10 min</div>
                    <div style="background: rgba(245, 158, 11, 0.2); color: {yawn_color}; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.75rem; display: inline-block; margin-top: 4px;">{yawn_status}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"""
                <div class="edge-card">
                    <div style="color: #94A3B8; font-size: 0.8rem; margin-bottom: 4px;">Head pose</div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 2px;"><span style="color:#94A3B8;">Pitch</span> <strong>{detector.pitch:+.0f}°</strong></div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 2px;"><span style="color:#94A3B8;">Yaw</span> <strong>{detector.yaw:+.0f}°</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#94A3B8;">Roll</span> <strong>{detector.roll:+.0f}°</strong></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown("<div class='edge-card'><div style='color:#94A3B8; font-size:0.8rem;'>Head pitch trend (degrees)</div></div>", unsafe_allow_html=True)
        if len(detector.pitch_history) > 1:
            xs = [p[0] for p in detector.pitch_history]
            ys = [p[1] for p in detector.pitch_history]
        else:
            xs = [0, 1, 2]
            ys = [detector.pitch, detector.pitch, detector.pitch]
        fig_p = px.line(x=xs, y=ys)
        fig_p.update_traces(line_color='#F59E0B', line_width=3)
        fig_p.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_p, use_container_width=True)

    yawn_reach = "WARNING (yawns limit reached)" if detector.yawn_count >= 3 else "NORMAL (within limits)"
    st.markdown(
        f"""
        <div class="edge-card">
            <div style="font-weight: 700; color: #FFFFFF;">Thresholds & Active Telemetry</div>
            <div style="color: #CBD5E1; font-size: 0.9rem; margin-top: 4px;">
                Yawn: 3 in 10 min &nbsp;|&nbsp; Head tilt threshold: 25° for 3 s &nbsp;|&nbsp; Current Status: <strong style="color: {'#EF4444' if detector.yawn_count >= 3 else '#10B981'};">{yawn_reach}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    render_edge_footer()

def render_fused_fatigue_score():
    render_edge_header("Fused Fatigue Score")
    detector = st.session_state.detector
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("<div class='edge-card' style='text-align: center;'>", unsafe_allow_html=True)
        # Semi-circular Gauge Chart using Plotly
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = float(detector.fused_score),
            title = {'text': "Fatigue Score", 'font': {'color': '#94A3B8', 'size': 16}},
            number = {'font': {'color': '#FFFFFF', 'size': 44, 'family': 'Segoe UI'}},
            gauge = {
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                'bar': {'color': "#FFFFFF", 'width': 3},
                'bgcolor': "rgba(0,0,0,0)",
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 0.4], 'color': '#10B981'},
                    {'range': [0.4, 0.7], 'color': '#F59E0B'},
                    {'range': [0.7, 1.0], 'color': '#EF4444'}
                ]
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        ear_score = max(0.0, min(1.0, (0.32 - detector.current_ear) / 0.15)) if detector.current_ear < 0.32 else 0.05
        yawn_score = min(1.0, detector.yawn_count / 4.0)
        head_score = min(1.0, abs(detector.pitch) / 30.0)
        hr_score = max(0.0, min(1.0, (75 - detector.heart_rate) / 25.0)) if detector.wearable_connected else 0.0
        
        st.markdown(
            f"""
            <div class="edge-card">
                <h3 style="color: #FFFFFF; margin-top: 0;">Signal contribution</h3>
                
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; margin-bottom: 4px;">
                        <span>Eye closure (EAR: {detector.current_ear:.2f}) (weight 35%)</span><strong>{ear_score:.2f}</strong>
                    </div>
                    <div style="background:#0B132B; border-radius:6px; height:8px;"><div style="background:{'#EF4444' if ear_score > 0.5 else '#10B981'}; width:{int(ear_score * 100)}%; height:100%; border-radius:6px;"></div></div>
                </div>
                
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; margin-bottom: 4px;">
                        <span>Yawning (Yawns: {detector.yawn_count}) (weight 15%)</span><strong>{yawn_score:.2f}</strong>
                    </div>
                    <div style="background:#0B132B; border-radius:6px; height:8px;"><div style="background:{'#F59E0B' if yawn_score > 0.3 else '#10B981'}; width:{int(yawn_score * 100)}%; height:100%; border-radius:6px;"></div></div>
                </div>
                
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; margin-bottom: 4px;">
                        <span>Head pose (Pitch: {detector.pitch:+.0f}°) (weight 15%)</span><strong>{head_score:.2f}</strong>
                    </div>
                    <div style="background:#0B132B; border-radius:6px; height:8px;"><div style="background:{'#EF4444' if head_score > 0.6 else '#10B981'}; width:{int(head_score * 100)}%; height:100%; border-radius:6px;"></div></div>
                </div>
                
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; margin-bottom: 4px;">
                        <span>Heart rate / HRV ({detector.heart_rate} bpm) (weight 20%)</span><strong>{hr_score:.2f}</strong>
                    </div>
                    <div style="background:#0B132B; border-radius:6px; height:8px;"><div style="background:#F59E0B; width:{int(hr_score * 100)}%; height:100%; border-radius:6px;"></div></div>
                </div>
                
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; margin-bottom: 4px;">
                        <span>Steering pattern (weight 15%)</span><strong>{detector.steering_variance:.2f}</strong>
                    </div>
                    <div style="background:#0B132B; border-radius:6px; height:8px;"><div style="background:#10B981; width:{int(detector.steering_variance * 100)}%; height:100%; border-radius:6px;"></div></div>
                </div>
                <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 15px;">Weights are dynamically distributed based on active sensor feeds.</div>
                
                <div style="background: {'rgba(239, 68, 68, 0.2)' if detector.fused_score >= 0.7 else ('rgba(245, 158, 11, 0.2)' if detector.fused_score >= 0.4 else 'rgba(16, 185, 129, 0.2)')}; color: {'#EF4444' if detector.fused_score >= 0.7 else ('#F59E0B' if detector.fused_score >= 0.4 else '#10B981')}; border: 1px solid {'#EF4444' if detector.fused_score >= 0.7 else ('#F59E0B' if detector.fused_score >= 0.4 else '#10B981')}; font-weight: 700; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; display: inline-block; margin-top: 10px;">
                    {'Critical Risk' if detector.fused_score >= 0.7 else ('Amber - Moderate Warning' if detector.fused_score >= 0.4 else 'Green - Normal Low Risk')}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    render_edge_footer()

def render_escalating_alerts():
    render_edge_header("Escalating Alerts")
    detector = st.session_state.detector
    
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            """
            <div class="edge-card" style="border-left: 6px solid #F59E0B;">
                <h3 style="color: #F59E0B; margin: 0;">STAGE 1</h3>
                <p style="color: #E2E8F0; margin-top: 4px;">Dashboard LED warning + soft chime</p>
            </div>
            
            <div style="text-align: center; color: #94A3B8; font-size: 0.8rem; margin: -6px 0 6px 0;">▼ escalates if unacknowledged</div>
            
            <div class="edge-card" style="border-left: 6px solid #F97316;">
                <h3 style="color: #F97316; margin: 0;">STAGE 2</h3>
                <p style="color: #E2E8F0; margin-top: 4px;">Buzzer + seat/steering vibration (no response within 10 s)</p>
            </div>
            
            <div style="text-align: center; color: #94A3B8; font-size: 0.8rem; margin: -6px 0 6px 0;">▼ escalates if unacknowledged</div>
            
            <div class="edge-card" style="border-left: 6px solid #EF4444;">
                <h3 style="color: #EF4444; margin: 0;">STAGE 3</h3>
                <p style="color: #E2E8F0; margin-top: 4px;">High-severity incident + GPS log + hazard lights (if installed)</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="edge-card">
                <div style="color: #FFFFFF; font-weight: 700; font-size: 1.1rem;">Alert status</div>
                <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 10px;">Current stage</div>
                <div style="font-size: 3rem; font-weight: 900; color: #F97316;">Stage 2</div>
                
                <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 10px;">Fatigue score</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #EF4444;">0.78</div>
                
                <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 10px;">Time unacknowledged</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #FFFFFF; margin-bottom: 20px;">00:07 s</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("ACKNOWLEDGE", type="primary", use_container_width=True):
            detector.acknowledge_alert()
            st.success("Alert Acknowledged. Stage Reset.")

    render_edge_footer()

def render_alert_configuration():
    render_edge_header("Alert Configuration")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='edge-card'>", unsafe_allow_html=True)
        
        tone = st.selectbox("Alert tone", ["Chime", "Voice Message"])
        volume = st.slider("Volume", 0, 100, 70, format="%d%%")
        vibration = st.select_slider("Vibration intensity", options=["Low", "Medium", "High"], value="Medium")
        escalation = st.slider("Escalation window", 3, 30, 10, format="%d s")
        voice_msg = st.toggle("Voice message", value=True)
        
        st.caption("<span style='color:#F59E0B;'>Stage 3 alert cannot be disabled (fleet policy)</span>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Save", type="primary", use_container_width=True):
            st.success("Alert Configuration Saved to Profile!")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    render_edge_footer()

def render_geotagged_incident_log():
    render_edge_header("Geo-Tagged Incident Log (Current Trip)")
    logger = EventLogger()
    df = logger.load_history()
    
    st.dataframe(df, use_container_width=True)
    
    st.markdown(
        """
        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 15px;">
            ● Incidents are stored locally and synced when connectivity is available.<br>
            ● Logs cannot be edited or deleted by the driver.
        </div>
        <br>
        <div style="display: flex; gap: 15px;">
            <div style="background: #1C2541; padding: 6px 16px; border-radius: 20px; font-weight: 700; color: #94A3B8; border: 1px solid #334155;">Pending sync: 2</div>
            <div style="background: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid #10B981; padding: 6px 16px; border-radius: 20px; font-weight: 700;">Last sync: 10:05</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    render_edge_footer()

def render_settings_grid():
    render_edge_header("Settings")
    
    tiles = [
        "Profile", "Change Password", "Alert Configuration",
        "Wearable Pairing", "Language", "Notifications",
        "Help / Report Issue", "Terms of Use", "Logout"
    ]
    
    rows = [tiles[0:3], tiles[3:6], tiles[6:9]]
    for r in rows:
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        for idx, title in enumerate(r):
            with cols[idx]:
                color_style = "color: #EF4444;" if title == "Logout" else ""
                st.markdown(f"""<div class="settings-tile" style="{color_style}">{title}</div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

    render_edge_footer()

def render_change_password():
    render_edge_header("Change Password")
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<div class='edge-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#FFFFFF; margin-top:0;'>Change Password</h3>", unsafe_allow_html=True)
        
        old_pwd = st.text_input("Old Password", type="password")
        new_pwd = st.text_input("New Password", type="password")
        confirm_pwd = st.text_input("Confirm New Password", type="password")
        
        st.caption("[Forgot password?](#)")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Update", type="primary", use_container_width=True):
            if not old_pwd or not new_pwd:
                st.error("Please fill all required password fields.")
            elif new_pwd != confirm_pwd:
                st.error("New Password and Confirm New Password must match.")
            else:
                st.success("Password Updated Successfully!")
                
        st.markdown("</div>", unsafe_allow_html=True)
        
    render_edge_footer()

def render_notifications_preferences():
    render_edge_header("Notifications")
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<div class='edge-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#FFFFFF; margin-top:0;'>Notifications</h3>", unsafe_allow_html=True)
        
        t1 = st.toggle("In-app alerts for high-severity incidents", value=True)
        t2 = st.toggle("Email alerts", value=True)
        t3 = st.toggle("SMS alerts", value=False)
        t4 = st.toggle("Trip summary at end of trip", value=True)
        
        st.caption("<span style='color:#F59E0B;'>At least one channel must stay enabled for high-severity incidents.</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    render_edge_footer()

def render_report_issue():
    render_edge_header("Report Issue / Help")
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<div class='edge-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#FFFFFF; margin-top:0;'>Report Issue / Help</h3>", unsafe_allow_html=True)
        
        issue_text = st.text_area("Describe your issue (max 200 words)", height=150)
        words = len(issue_text.split()) if issue_text else 0
        st.caption(f"Words: {words} / 200")
        
        if st.button("Send", type="primary", use_container_width=True):
            if not issue_text:
                st.error("Issue description is mandatory.")
            elif words > 200:
                st.error("User is allowed to explain the issue in 200 words only.")
            else:
                st.success("Issue submitted to support team!")
                
        st.markdown("</div>", unsafe_allow_html=True)
        
    render_edge_footer()

def render_terms_of_use():
    render_edge_header("Terms of Use")
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            """
            <div class="edge-card">
                <h3 style="color: #FFFFFF; margin-top: 0;">Terms of Use & Privacy Policy</h3>
                <ol style="color: #CBD5E1; line-height: 1.8;">
                    <li>Camera data is processed on-device; raw video is not stored.</li>
                    <li>Only derived metrics and incident records are uploaded.</li>
                    <li>Data shared with insurers only with your consent.</li>
                    <li>Alerts are an aid and do not replace safe driving.</li>
                    <li>You may request deletion of your data at any time.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    render_edge_footer()

def render_logout_confirmation():
    render_edge_header("Logout")
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        current_drv = st.session_state.get("current_driver", {})
        is_logged = st.session_state.get("logged_in", False)
        
        st.markdown(
            f"""
            <div class="edge-card" style="text-align: center; padding: 40px 20px;">
                <h2 style="color: #FFFFFF; margin: 0;">Log out Driver?</h2>
                <p style="color: #94A3B8; margin: 15px 0 10px 0;">Active Driver: <strong style="color:#00B4D8;">{current_drv.get('name', 'None')}</strong> ({current_drv.get('driver_id', 'N/A')})</p>
                <p style="color: #64748B; font-size: 0.85rem; margin-bottom: 25px;">Vehicle: {current_drv.get('vehicle_id', 'N/A')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        b_cancel, b_logout = st.columns(2)
        with b_cancel:
            st.button("Cancel", use_container_width=True)
        with b_logout:
            if st.button("Logout", type="primary", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.current_driver = {
                    "driver_id": "LOGGED_OUT",
                    "name": "Guest (Logged Out)",
                    "vehicle_id": "N/A",
                    "shift": "None",
                    "email": ""
                }
                st.success("Logged out successfully! Please navigate to '4.1 Driver Login' to sign in.")
                time.sleep(0.5)
                st.rerun()

    render_edge_footer()


# ==============================================================================
# FLEET & CLOUD DASHBOARD MODULES (WEB PLATFORM)
# ==============================================================================

def render_fleet_home():
    render_browser_header("dashboard")
    
    st.markdown("## Fleet Dashboard")
    
    # 5 KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown("""<div class="fleet-kpi-card"><div class="kpi-title">Vehicles online</div><div class="kpi-value">48 <span style="font-size:1.2rem; color:#64748B;">/ 52</span></div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown("""<div class="fleet-kpi-card"><div class="kpi-title">Trips today</div><div class="kpi-value">136</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown("""<div class="fleet-kpi-card"><div class="kpi-title">Incidents today</div><div class="kpi-value" style="color:#F59E0B;">23</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown("""<div class="fleet-kpi-card"><div class="kpi-title">High severity</div><div class="kpi-value" style="color:#EF4444;">4</div></div>""", unsafe_allow_html=True)
    with k5:
        st.markdown("""<div class="fleet-kpi-card"><div class="kpi-title">Drivers at risk</div><div class="kpi-value" style="color:#EF4444;">6</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2 Charts
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        st.markdown("### Incidents per day (last 7 days)")
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        incidents = [12, 16, 10, 19, 14, 21, 19]
        fig1 = px.bar(x=days, y=incidents, labels={'x':'Day', 'y':'Incidents'})
        fig1.update_traces(marker_color='#1E40AF')
        fig1.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='#FFFFFF')
        st.plotly_chart(fig1, use_container_width=True)

    with c_chart2:
        st.markdown("### Incidents by time of day")
        hours = ["0-4", "4-8", "8-12", "12-16", "16-20", "20-24"]
        counts = [8, 4, 11, 18, 7, 4]
        fig2 = px.bar(x=hours, y=counts, labels={'x':'Time Slot', 'y':'Count'})
        fig2.update_traces(marker_color='#F59E0B')
        fig2.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='#FFFFFF')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Latest incidents")
    inc_df = st.session_state.incidents_df
    st.dataframe(inc_df[["time", "driver", "vehicle", "score", "severity", "status"]], use_container_width=True)

def render_create_driver():
    render_browser_header("drivers/create")
    
    st.markdown("## Create Driver Account")
    
    c_form, c_rules = st.columns([3, 2])
    
    with c_form:
        with st.form("create_driver_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Driver Name *", placeholder="e.g. Ramesh Kumar")
                email = st.text_input("Email ID *", placeholder="e.g. ramesh@fleet.in")
                password = st.text_input("Password (min 6 chars) *", type="password")
            with col2:
                driver_id = st.text_input("Driver ID (e.g. DRV2026R) *", placeholder="DRV2026...")
                phone = st.text_input("Phone Number * (10 digits)", placeholder="9876543210")
                confirm_password = st.text_input("Confirm Password *", type="password")
                
            c_v1, c_v2 = st.columns(2)
            with c_v1:
                vehicle_id = st.text_input("Assigned Vehicle ID (Optional)", placeholder="e.g. UP32-AB-9999")
            with c_v2:
                shift = st.selectbox("Shift Type", ["Day (8 AM – 5 PM)", "Night (10 PM – 6 AM)", "Rotating"])
                
            submitted = st.form_submit_button("Create Driver", type="primary")
            if submitted:
                if password != confirm_password:
                    st.error("Passwords do not match. Please re-enter.")
                else:
                    success, res = register_driver(
                        name=name,
                        email=email,
                        password=password,
                        driver_id=driver_id,
                        phone=phone,
                        vehicle_id=vehicle_id,
                        shift=shift
                    )
                    if success:
                        st.session_state.drivers_list = load_mock_drivers()
                        st.success(f"🎉 Driver '{name}' ({driver_id.upper()}) created and saved to database successfully!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(f"❌ {res}")

    with c_rules:
        st.markdown(
            """
            <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 12px; padding: 20px; color: #1E3A8A;">
                <h4 style="margin-top:0;">Driver Registration Rules</h4>
                <div style="margin-bottom: 6px;">✓ Unique Driver ID & Email required</div>
                <div style="margin-bottom: 6px;">✓ Minimum 6 characters password</div>
                <div style="margin-bottom: 6px;">✓ Valid 10-digit phone number</div>
                <div style="margin-bottom: 6px;">✓ Instant persistence to <code>data/drivers.csv</code></div>
                <div>✓ Driver can immediately log in from the Edge Unit</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Registered Fleet Drivers Database")
    drivers_df = load_drivers_df()
    display_cols = ["driver_id", "name", "email", "phone", "vehicle_id", "shift", "calibration_status"]
    available_cols = [c for c in display_cols if c in drivers_df.columns]
    st.dataframe(drivers_df[available_cols], use_container_width=True)

def render_driver_profile_assignment():
    render_browser_header("drivers/profile")
    
    st.markdown("## Driver Profile & Vehicle Assignment")
    
    drivers = st.session_state.drivers_list
    driver_names = [d["name"] for d in drivers] if drivers else ["Rahul Verma"]
    
    current_name = st.session_state.current_driver.get("name", driver_names[0])
    default_idx = driver_names.index(current_name) if current_name in driver_names else 0
    
    selected_name = st.selectbox("Select Driver to View / Edit Profile", driver_names, index=default_idx)
    selected_driver = next((d for d in drivers if d["name"] == selected_name), st.session_state.current_driver)
    
    name_parts = selected_driver.get("name", "").split(" ", 1)
    first_name = name_parts[0]
    last_name = name_parts[1] if len(name_parts) > 1 else ""
    
    c_photo, c_info = st.columns([1, 4])
    with c_photo:
        st.markdown("<div style='width:100px; height:100px; border-radius:50%; background:#CBD5E1; display:flex; align-items:center; justify-content:center; font-size:3rem;'>👤</div>", unsafe_allow_html=True)
        st.file_uploader("Profile photo (optional)", type=["png", "jpg"])
        
    with c_info:
        c1, c2 = st.columns(2)
        with c1:
            fn = st.text_input("First Name", value=first_name)
            dob = st.text_input("Date of Birth", value=str(selected_driver.get("dob", "14 / 08 / 1994")))
            shift = st.text_input("Shift Type", value=str(selected_driver.get("shift", "Night (10 PM – 6 AM)")))
            phone = st.text_input("Phone / Contact", value=str(selected_driver.get("phone", "9876543210")))
        with c2:
            ln = st.text_input("Last Name", value=last_name)
            lic = st.text_input("Licence Number", value=str(selected_driver.get("licence_no", "UP32 20190012345")))
            veh = st.text_input("Assigned Vehicle ID", value=str(selected_driver.get("vehicle_id", "UP32-AB-1234")))
            calib = st.text_input("Calibration Status", value=str(selected_driver.get("calibration_status", "Completed")), disabled=True)
            
        if st.button("Save Profile & Assignment", type="primary"):
            updated_name = f"{fn.strip()} {ln.strip()}".strip()
            up_dict = {
                "name": updated_name,
                "dob": dob.strip(),
                "shift": shift.strip(),
                "phone": phone.strip(),
                "licence_no": lic.strip(),
                "vehicle_id": veh.strip()
            }
            drv_id = selected_driver.get("driver_id", "")
            success, msg = update_driver_profile(drv_id, up_dict)
            if success:
                st.session_state.drivers_list = load_mock_drivers()
                if st.session_state.current_driver.get("driver_id") == drv_id:
                    st.session_state.current_driver.update(up_dict)
                st.success(f"Profile details for {updated_name} updated successfully.")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(msg)

def render_driver_scorecards():
    render_browser_header("drivers")
    
    st.markdown("## Driver Scorecards")
    
    c_list, c_details = st.columns([2, 3])
    drivers = st.session_state.drivers_list
    
    with c_list:
        search_query = st.text_input("🔍 Search driver name / ID", value="").strip().lower()
        if search_query:
            filtered_drivers = [d for d in drivers if search_query in str(d.get("name","")).lower() or search_query in str(d.get("driver_id","")).lower()]
        else:
            filtered_drivers = drivers
            
        if not filtered_drivers:
            st.info("No drivers match your search.")
            selected_d = drivers[0] if drivers else {}
        else:
            driver_choices = [f"{d['name']} • {d['driver_id']}" for d in filtered_drivers]
            sel_idx = st.radio("Select Driver to Inspect:", range(len(driver_choices)), format_func=lambda x: driver_choices[x])
            selected_d = filtered_drivers[sel_idx]

    with c_details:
        if selected_d:
            st.markdown(f"### {selected_d.get('name', 'Driver')} • {selected_d.get('driver_id', 'N/A')}")
            st.caption(f"Vehicle: **{selected_d.get('vehicle_id', 'N/A')}** | Shift: **{selected_d.get('shift', 'N/A')}**")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trips", str(selected_d.get("trips", 0)))
            m2.metric("Incidents", str(selected_d.get("incidents", 0)))
            m3.metric("Avg score", f"{float(selected_d.get('avg_score', 0.20)):.2f}")
            m4.metric("High sev.", str(selected_d.get("high_sev", 0)))
            
            st.markdown("#### 30-day fatigue trend")
            x_trend = np.linspace(1, 30, 10)
            base_score = float(selected_d.get("avg_score", 0.3))
            y_trend = [round(max(0.1, min(0.9, base_score + 0.1 * np.sin(i))), 2) for i in x_trend]
            fig_tr = px.line(x=x_trend, y=y_trend, labels={'x': 'Days in Month', 'y': 'Fatigue Score'})
            fig_tr.update_traces(line_color='#1E40AF', line_width=3)
            fig_tr.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_tr, use_container_width=True)

            sev = selected_d.get("severity_breakdown", {})
            if not isinstance(sev, dict):
                sev = {"Low": 1, "Medium": 0, "High": 0}
            st.markdown("#### Severity breakdown")
            st.markdown(f"""
            <div style="margin-bottom:6px;">Low &nbsp;<span style="background:#10B981; padding:2px 30px; border-radius:4px; color:white; font-weight:700;">{sev.get('Low', 0)}</span></div>
            <div style="margin-bottom:6px;">Medium &nbsp;<span style="background:#F59E0B; padding:2px 30px; border-radius:4px; color:white; font-weight:700;">{sev.get('Medium', 0)}</span></div>
            <div>High &nbsp;<span style="background:#EF4444; padding:2px 30px; border-radius:4px; color:white; font-weight:700;">{sev.get('High', 0)}</span></div>
            """, unsafe_allow_html=True)

def render_incident_review():
    render_browser_header("incidents")
    
    st.markdown("## Incident Review Workflow")
    
    drivers = st.session_state.drivers_list
    driver_names = ["All Drivers"] + [d["name"] for d in drivers]
    vehicle_ids = ["All Vehicles"] + list(dict.fromkeys([d["vehicle_id"] for d in drivers if d.get("vehicle_id")]))
    
    f1, f2, f3, f4 = st.columns(4)
    f1.selectbox("Vehicle", vehicle_ids)
    f2.selectbox("Driver", driver_names)
    f3.selectbox("Severity", ["All Severities", "High", "Medium", "Low"])
    f4.selectbox("Status", ["All Status", "Open", "Under Review", "Closed"])
    
    c_table, c_detail = st.columns([3, 2])
    
    with c_table:
        inc_df = st.session_state.incidents_df
        st.dataframe(inc_df[["time", "driver", "severity", "status"]], use_container_width=True)

    with c_detail:
        st.markdown("<div style='background:#FFFFFF; border:1px solid #CBD5E1; border-radius:12px; padding:16px;'>", unsafe_allow_html=True)
        st.markdown("### Incident #INC-2041")
        
        # Route Map Graphic
        fig_map = px.scatter(x=[80.9462], y=[26.8467], size=[20], color_discrete_sequence=['#EF4444'])
        fig_map.update_layout(height=140, margin=dict(l=0, r=0, t=0, b=0), xaxis_visible=False, yaxis_visible=False)
        st.plotly_chart(fig_map, use_container_width=True)
        
        st.markdown("""
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;"><span>Fatigue score</span><strong>0.78</strong></div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;"><span>Alert stage</span><strong>2</strong></div>
        <div style="display:flex; justify-content:space-between; margin-bottom:12px;"><span>Driver response</span><strong>Acknowledged</strong></div>
        """, unsafe_allow_html=True)
        
        remarks = st.text_area("Remarks (mandatory, max 300 chars)", value="Driver advised to rest at next stop.", height=80)
        
        b_add, b_close = st.columns(2)
        with b_add:
            st.button("Add Remark", use_container_width=True)
        with b_close:
            if st.button("Close Incident", type="primary", use_container_width=True):
                st.success("Incident Marked as Closed.")
                
        st.markdown("</div>", unsafe_allow_html=True)

def render_threshold_configuration():
    render_browser_header("settings")
    
    st.markdown("## Threshold Configuration")
    
    c1, c2 = st.columns([3, 2])
    
    with c1:
        st.markdown("### Fleet-wide defaults")
        col_a, col_b = st.columns(2)
        with col_a:
            st.number_input("Eye-closure time (s)", value=2.0, min_value=0.5, max_value=5.0)
            st.number_input("Head-pose angle (°)", value=25, min_value=10, max_value=45)
            st.number_input("Fatigue score - Critical", value=0.75, min_value=0.1, max_value=1.0)
        with col_b:
            st.number_input("Yawn count (per 10 min)", value=3, min_value=1, max_value=10)
            st.number_input("Fatigue score - Warning", value=0.50, min_value=0.1, max_value=1.0)
            st.number_input("Escalation window (s)", value=10, min_value=3, max_value=30)
            
        b_save, b_reset = st.columns(2)
        with b_save:
            st.button("Save Thresholds", type="primary", use_container_width=True)
        with b_reset:
            st.button("Reset to default", use_container_width=True)

    with c2:
        st.markdown("### Per-driver override")
        st.selectbox("Driver", ["Rahul Verma", "Amit Singh", "Sunil Yadav"])
        st.number_input("Driver Eye-closure time (s)", value=1.5)
        st.toggle("Enable override", value=True)
        
        st.markdown("#### Change log")
        st.caption("09 Oct - Priya: score 0.75\n02 Oct - Admin: yawns 3")

def render_fatigue_risk_prediction():
    render_browser_header("analytics")
    
    st.markdown("## Fatigue-Risk Prediction")
    
    c_plan, c_result = st.columns([1, 1])
    
    with c_plan:
        st.markdown("### Plan a trip")
        st.selectbox("Select Driver", ["Rahul Verma", "Amit Singh"])
        st.text_input("Planned start", value="10 Oct 2026, 22:00")
        st.text_input("Expected duration", value="9 hours")
        st.selectbox("Route", ["Lucknow → Delhi", "Lucknow → Kanpur", "Kanpur → Prayagraj"])
        
        if st.button("Predict Risk", type="primary", use_container_width=True):
            st.info("Computing Predictive ML Risk Index...")

    with c_result:
        st.markdown(
            """
            <div style="background:#FEF2F2; border:2px solid #EF4444; border-radius:12px; padding:20px; text-align:center;">
                <div style="font-size:0.9rem; color:#991B1B; font-weight:700;">Predicted fatigue risk</div>
                <div style="font-size:3.5rem; font-weight:900; color:#DC2626;">HIGH</div>
                <div style="font-size:1.1rem; color:#7F1D1D;">Risk score 0.81</div>
                <div style="background:#DC2626; color:white; padding:4px 14px; border-radius:20px; font-size:0.8rem; font-weight:700; display:inline-block; margin-top:8px;">Dispatcher action needed</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("#### Contributing factors")
        st.markdown("""
        <div>Night shift (22:00–07:00) <span style="color:#EF4444; float:right;">90%</span></div>
        <div style="background:#E2E8F0; border-radius:4px; height:8px; margin-bottom:8px;"><div style="background:#EF4444; width:90%; height:100%; border-radius:4px;"></div></div>
        
        <div>Shift length > 8 hrs <span style="color:#EF4444; float:right;">85%</span></div>
        <div style="background:#E2E8F0; border-radius:4px; height:8px; margin-bottom:8px;"><div style="background:#EF4444; width:85%; height:100%; border-radius:4px;"></div></div>
        
        <div>Past incident pattern <span style="color:#F59E0B; float:right;">50%</span></div>
        <div style="background:#E2E8F0; border-radius:4px; height:8px; margin-bottom:8px;"><div style="background:#F59E0B; width:50%; height:100%; border-radius:4px;"></div></div>
        """, unsafe_allow_html=True)
        
        st.warning("Recommendation: schedule a rest break or change driver.")

def render_route_analytics():
    render_browser_header("routes")
    
    st.markdown("## Route-Level Risk Analytics")
    
    c_map, c_rank = st.columns([3, 2])
    
    with c_map:
        st.markdown("### Route Incident Heatmap")
        fig_route = px.scatter(
            x=[80.94, 80.33, 78.00, 81.84],
            y=[26.84, 26.44, 27.17, 25.43],
            size=[40, 30, 25, 15],
            text=["Lucknow", "Kanpur", "Agra", "Prayagraj"],
            color_discrete_sequence=['#DC2626']
        )
        fig_route.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_route, use_container_width=True)

    with c_rank:
        st.markdown("### Highest-risk routes")
        routes_data = [
            {"route": "Lucknow - Kanpur NH27", "rate": "9.2"},
            {"route": "Agra Expy km 120-180", "rate": "8.4"},
            {"route": "Lucknow - Sitapur", "rate": "6.1"},
            {"route": "Kanpur - Prayagraj", "rate": "5.7"}
        ]
        for r in routes_data:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:10px 14px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;">
                <div><strong>{r['route']}</strong><br><span style="font-size:0.75rem; color:#64748B;">incidents / 100 trips</span></div>
                <div style="font-size:1.4rem; font-weight:900; color:#DC2626;">{r['rate']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("#### Peak time slots")
        fig_peak = px.bar(x=["0", "4", "8", "12", "16", "20"], y=[2, 4, 8, 12, 6, 3])
        fig_peak.update_traces(marker_color='#F59E0B')
        fig_peak.update_layout(height=120, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_peak, use_container_width=True)

def render_reports_export():
    render_browser_header("reports")
    
    st.markdown("## Reports & Data Export")
    
    c_gen, c_recent = st.columns([3, 2])
    
    with c_gen:
        st.markdown("### Generate report")
        report_type = st.selectbox("Report type", ["Incident log", "Driver scorecard", "Route risk"])
        
        c_from, c_to = st.columns(2)
        with c_from:
            from_date = st.date_input("From date", value=datetime.date(2026, 9, 1))
        with c_to:
            to_date = st.date_input("To date", value=datetime.date(2026, 9, 30))
            
        fmt = st.radio("Format", ["CSV", "Excel", "PDF"], horizontal=True)
        st.caption("Maximum export range: 12 months")
        
        if st.button("Download", type="primary", use_container_width=True):
            fname, fpath = generate_report_file(report_type, str(from_date), str(to_date), fmt)
            st.success(f"Report Generated: {fname}")
            with open(fpath, "rb") as file_bytes:
                st.download_button("Click to Save File", data=file_bytes, file_name=fname, mime="application/octet-stream")

    with c_recent:
        st.markdown("### Recent exports")
        exports = ["incidents_sep26.xlsx", "scorecards_aug26.pdf", "route_risk_q2.csv"]
        for exp in exports:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:12px; margin-bottom:8px; font-family:monospace; color:#334155;">
                📄 {exp}
            </div>
            """, unsafe_allow_html=True)

def render_fleet_login():
    render_browser_header("login")
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:12px; padding:30px; text-align:center;">
                <h1 style="color:#0F172A; margin:0;">DrowsiGuard</h1>
                <p style="color:#64748B;">Fleet Safety Platform</p>
                
                <h3 style="margin-top:20px;">Fleet Manager Login</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        email = st.text_input("Email ID", value="manager@fleet.in")
        pwd = st.text_input("Password", value="••••••••••", type="password")
        st.caption("[Forgot Password?](#)")
        
        if st.button("Login to Fleet Platform", type="primary", use_container_width=True):
            st.success("Login Successful! Welcome Fleet Manager.")
            
        st.caption("Session expires after 30 minutes of inactivity")


if __name__ == "__main__":
    main()
