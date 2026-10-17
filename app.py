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
    generate_report_file
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
if "current_driver" not in st.session_state:
    st.session_state.current_driver = DEFAULT_DRIVER.copy()
if "incidents_df" not in st.session_state:
    st.session_state.incidents_df = load_mock_incidents()
if "drivers_list" not in st.session_state:
    st.session_state.drivers_list = load_mock_drivers()
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
                "3. Splash Screen",
                "4.1 Driver Login",
                "4.2 Forgot Password",
                "6.1 Camera & Driver Calibration",
                "6.2 Live Monitoring Screen",
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
            ]
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
        st.markdown(
            """
            <div style="background: #1C2541; padding: 30px; border-radius: 16px; border: 1px solid #334155; text-align: center;">
                <h2 style="color: #FFFFFF; margin: 0;">Welcome Back, Driver</h2>
                <p style="color: #94A3B8;">Login to start your trip</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        with st.form("driver_login_form"):
            driver_id = st.text_input("Driver ID / Email", value="rahul.verma@fleet.in", max_chars=30)
            password = st.text_input("Password", value="••••••••••", type="password", max_chars=20)
            
            c_forgot, c_sub = st.columns([1, 1])
            with c_forgot:
                st.caption("[Forgot Password?](#)")
            with c_sub:
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
                
            if submitted:
                if not driver_id:
                    st.error("Please Enter Driver ID or Email")
                elif not password:
                    st.error("Please Enter Password")
                elif driver_id != "rahul.verma@fleet.in":
                    st.error("The Driver ID or Email is not Registered")
                else:
                    st.success("Login Successful! Redirecting to Calibration...")
                    
        st.caption("Max 30 characters • Password max 20 characters")
        
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
    
    col_cam, col_calib = st.columns([3, 2])
    
    with col_cam:
        st.markdown(
            """
            <div style="position: relative; background: #000; border-radius: 12px; overflow: hidden; border: 2px solid #10B981; height: 380px; display: flex; align-items: center; justify-content: center;">
                <div style="position: absolute; top: 15px; left: 15px; color: #EF4444; font-weight: 700;">● LIVE</div>
                <div style="position: absolute; top: 15px; right: 15px; color: #10B981; font-weight: 700;">Face Found</div>
                <!-- Mock Face Calibration Oval -->
                <div style="border: 3px solid #10B981; width: 180px; height: 230px; border-radius: 50%; display: flex; align-items: center; justify-content: center; background: rgba(16, 185, 129, 0.05);">
                    <div style="font-size: 3rem;">👤</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_calib:
        st.markdown(
            """
            <div class="edge-card">
                <h3 style="color: #FFFFFF; margin-top: 0;">Calibration</h3>
                <div style="color: #10B981; font-weight: 700; margin-bottom: 8px;">🟢 Face Found</div>
                <div style="color: #10B981; font-weight: 700; margin-bottom: 8px;">🟢 Eyes Detected</div>
                <div style="color: #F59E0B; font-weight: 700; margin-bottom: 8px;">🟠 Baseline EAR captured</div>
                <div style="color: #10B981; font-weight: 700; margin-bottom: 16px;">🟢 Lighting OK</div>
                
                <div style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase;">Baseline EAR</div>
                <div style="font-size: 2.8rem; font-weight: 900; color: #00B4D8;">0.31</div>
                
                <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 10px;">Calibrating... 65%</div>
                <div style="background-color: #0B132B; border-radius: 8px; height: 10px; margin-bottom: 20px;">
                    <div style="background-color: #00B4D8; width: 65%; height: 100%; border-radius: 8px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.button("Start Calibration", type="primary", use_container_width=True)
        st.button("Skip (use last calibration)", use_container_width=True)
        
    render_edge_footer()

def render_live_monitoring():
    render_edge_header("Live Monitoring")
    detector = st.session_state.detector
    
    col_left, col_right = st.columns([3, 3])
    
    with col_left:
        st.markdown("#### 📹 Live Video Stream")
        
        # Camera Mode Selector
        cam_source = st.radio("Camera Feed Source", ["🌐 WebRTC Browser Camera", "📹 Direct OpenCV Webcam"], horizontal=True)
        
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
        else: # Direct OpenCV Webcam
            st.info("Capturing live video feed directly from system webcam (OpenCV VideoCapture)...")
            frame_placeholder = st.empty()
            col_start, col_stop = st.columns(2)
            with col_start:
                start_btn = st.button("Start Live OpenCV Stream", type="primary", use_container_width=True)
            with col_stop:
                stop_btn = st.button("Stop Stream", use_container_width=True)
                
            if start_btn:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    st.success("OpenCV Camera Feed Active! Streaming in real-time...")
                    while cap.isOpened() and not stop_btn:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Failed to grab camera frame.")
                            break
                        annotated = detector.process_frame(frame)
                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        time.sleep(0.01)
                    cap.release()
                else:
                    st.error("Hardware Camera (OpenCV Index 0) not accessible or currently in use by another app.")


        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="edge-card">
                <div style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase;">Fatigue Score (fused)</div>
                <div style="background-color: #0B132B; border-radius: 8px; height: 14px; margin: 8px 0;">
                    <div style="background-color: #10B981; width: {int(detector.fused_score * 100)}%; height: 100%; border-radius: 8px;"></div>
                </div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #10B981;">
                    {detector.fused_score} • Low risk
                </div>
                <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">
                    Signals: Eye ✓ &nbsp; Yawn ✓ &nbsp; Head ✓ &nbsp; HR ✓ &nbsp; Steering ✓
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_right:
        # Top Driver Status Card
        status_class = "status-alert" if not detector.is_drowsy else "status-drowsy"
        status_text = "ALERT" if not detector.is_drowsy else "DROWSY"
        
        st.markdown(
            f"""
            <div class="edge-card" style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="color: #94A3B8; font-size: 0.8rem; text-transform: uppercase;">Driver Status</div>
                    <div class="{status_class}" style="display: inline-block; margin-top: 4px;">{status_text}</div>
                </div>
                <div style="text-align: right;">
                    <div style="background: #0B132B; padding: 6px 12px; border-radius: 20px; font-weight: 700; color: #E2E8F0; font-size: 0.9rem;">Trip 00:42:18</div>
                    <div style="background: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid #10B981; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; margin-top: 8px; display: inline-block;">Calibrated</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 6 Telemetry Metric Cards Grid
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="edge-card"><div class="edge-metric-label">EAR</div><div class="edge-metric-val">{detector.current_ear}</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="edge-card"><div class="edge-metric-label">Blink rate</div><div class="edge-metric-val">{detector.blink_count}<span style="font-size:0.9rem; font-weight:400;">/min</span></div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="edge-card"><div class="edge-metric-label">Eye closure</div><div class="edge-metric-val">0.2<span style="font-size:0.9rem; font-weight:400;"> s</span></div></div>""", unsafe_allow_html=True)
            
        m4, m5, m6 = st.columns(3)
        with m4:
            st.markdown(f"""<div class="edge-card"><div class="edge-metric-label">Yawns</div><div class="edge-metric-val">{detector.yawn_count}</div></div>""", unsafe_allow_html=True)
        with m5:
            st.markdown(f"""<div class="edge-card"><div class="edge-metric-label">Head pose</div><div class="edge-metric-val">{abs(int(detector.pitch))}°</div></div>""", unsafe_allow_html=True)
        with m6:
            st.markdown(f"""<div class="edge-card"><div class="edge-metric-label">Heart rate</div><div class="edge-metric-val">{detector.heart_rate}<span style="font-size:0.9rem; font-weight:400;">bpm</span></div></div>""", unsafe_allow_html=True)

        # Action Buttons
        b1, b2, b3 = st.columns([1, 1, 1])
        with b1:
            st.button("Start Trip", type="primary", use_container_width=True)
        with b2:
            st.button("End Trip", use_container_width=True)
        with b3:
            st.button("Mute Chime", use_container_width=True)
            
        # Recent Events Box
        st.markdown(
            """
            <div class="edge-card">
                <div style="color: #94A3B8; font-size: 0.85rem; font-weight: 700; margin-bottom: 8px;">Recent events</div>
                <div style="font-size: 0.85rem; color: #CBD5E1; margin-bottom: 4px;">10:31 Yawn detected</div>
                <div style="font-size: 0.85rem; color: #CBD5E1;">10:12 Blink rate low — Warning cleared</div>
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
            """
            <div style="position: relative; background: #000; border-radius: 12px; overflow: hidden; border: 2px solid #00B4D8; height: 350px; display: flex; align-items: center; justify-content: center;">
                <div style="position: absolute; top: 15px; left: 15px; color: #EF4444; font-weight: 700;">● LIVE</div>
                <div style="position: absolute; top: 15px; right: 15px; color: #10B981; font-weight: 700;">Tracking</div>
                <div style="text-align: center;">
                    <div style="font-size: 4rem;">👤</div>
                    <div style="color: #F59E0B; font-weight: 700; margin-top: 10px;">MAR 0.62</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_metrics:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                <div class="edge-card">
                    <div style="color: #94A3B8; font-size: 0.8rem;">Yawn count (trip)</div>
                    <div style="font-size: 3rem; font-weight: 900; color: #F59E0B;">3</div>
                    <div style="font-size: 0.75rem; color: #94A3B8;">Limit: 3 in 10 min</div>
                    <div style="background: rgba(245, 158, 11, 0.2); color: #F59E0B; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.75rem; display: inline-block; margin-top: 4px;">WARNING</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                """
                <div class="edge-card">
                    <div style="color: #94A3B8; font-size: 0.8rem; margin-bottom: 4px;">Head pose</div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 2px;"><span style="color:#94A3B8;">Pitch</span> <strong>18°</strong></div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 2px;"><span style="color:#94A3B8;">Yaw</span> <strong>5°</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#94A3B8;">Roll</span> <strong>3°</strong></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown("<div class='edge-card'><div style='color:#94A3B8; font-size:0.8rem;'>Head pitch over time (degrees)</div></div>", unsafe_allow_html=True)
        x_pitch = np.linspace(0, 10, 20)
        y_pitch = [10, 11, 12, 14, 16, 18, 20, 22, 21]
        fig_p = px.line(x=x_pitch[:len(y_pitch)], y=y_pitch)
        fig_p.update_traces(line_color='#F59E0B', line_width=3)
        fig_p.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown(
        """
        <div class="edge-card">
            <div style="font-weight: 700; color: #FFFFFF;">Thresholds</div>
            <div style="color: #CBD5E1; font-size: 0.9rem; margin-top: 4px;">
                Yawn: 3 in 10 min &nbsp;|&nbsp; Head tilt: 25° for 3 s &nbsp;|&nbsp; Status: <strong style="color: #F59E0B;">WARNING (yawns reached limit)</strong>
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
            value = 0.58,
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
        st.markdown(
            """
            <div class="edge-card">
                <h3 style="color: #FFFFFF; margin-top: 0;">Signal contribution</h3>
                
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; margin-bottom: 4px;">
                        <span>Eye closure (EAR) (weight 35%)</span><strong>0.62</strong>
                    </div>
                    <div style="background:#0B132B; border-radius:6px; height:8px;"><div style="background:#F59E0B; width:62%; height:100%; border-radius:6px;"></div></div>
                </div>
                
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; margin-bottom: 4px;">
                        <span>Yawning (weight 15%)</span><strong>0.70</strong>
                    </div>
                    <div style="background:#0B132B; border-radius:6px; height:8px;"><div style="background:#F59E0B; width:70%; height:100%; border-radius:6px;"></div></div>
                </div>
                
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; margin-bottom: 4px;">
                        <span>Head pose (weight 15%)</span><strong>0.40</strong>
                    </div>
                    <div style="background:#0B132B; border-radius:6px; height:8px;"><div style="background:#10B981; width:40%; height:100%; border-radius:6px;"></div></div>
                </div>
                
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; margin-bottom: 4px;">
                        <span>Heart rate / HRV (weight 20%)</span><strong>0.55</strong>
                    </div>
                    <div style="background:#0B132B; border-radius:6px; height:8px;"><div style="background:#F59E0B; width:55%; height:100%; border-radius:6px;"></div></div>
                </div>
                
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #CBD5E1; margin-bottom: 4px;">
                        <span>Steering pattern (weight 15%)</span><strong>0.50</strong>
                    </div>
                    <div style="background:#0B132B; border-radius:6px; height:8px;"><div style="background:#F59E0B; width:50%; height:100%; border-radius:6px;"></div></div>
                </div>
                
                <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 15px;">Weights are redistributed if a signal is unavailable.</div>
                
                <div style="background: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid #F59E0B; font-weight: 700; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; display: inline-block; margin-top: 10px;">Amber - Warning</div>
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
        st.markdown(
            """
            <div class="edge-card" style="text-align: center; padding: 40px 20px;">
                <h2 style="color: #FFFFFF; margin: 0;">Log out?</h2>
                <p style="color: #94A3B8; margin: 15px 0 30px 0;">End the current trip before logging out.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        b_cancel, b_logout = st.columns(2)
        with b_cancel:
            st.button("Cancel", use_container_width=True)
        with b_logout:
            if st.button("Logout", type="primary", use_container_width=True):
                st.warning("Trip must be ended before logout!")

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
                name = st.text_input("Driver Name", value="Rahul Verma")
                email = st.text_input("Email ID", value="rahul.verma@fleet.in")
                password = st.text_input("Password (8-20 characters)", value="••••••••••", type="password")
            with col2:
                driver_id = st.text_input("Driver ID (min 6, alphanumeric)", value="DRV2026A")
                phone = st.text_input("Phone Number", value="9876543210")
                confirm_password = st.text_input("Confirm Password", value="••••••••••", type="password")
                
            submitted = st.form_submit_button("Create Driver", type="primary")
            if submitted:
                if len(driver_id) < 6 or not driver_id.isalnum():
                    st.error("Driver ID must be alphanumeric and a minimum of 6 characters.")
                elif len(phone) != 10 or not phone.isdigit():
                    st.error("Phone number must be a valid 10-digit number.")
                else:
                    st.success("Driver Account Created Successfully!")

    with c_rules:
        st.markdown(
            """
            <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 12px; padding: 20px; color: #1E3A8A;">
                <h4 style="margin-top:0;">Password rules</h4>
                <div style="margin-bottom: 6px;">✓ 8–20 characters</div>
                <div style="margin-bottom: 6px;">✓ 1 uppercase letter</div>
                <div style="margin-bottom: 6px;">✓ 1 lowercase letter</div>
                <div style="margin-bottom: 6px;">✓ 1 numeric digit</div>
                <div style="margin-bottom: 6px;">✓ 1 special character</div>
                <div>✓ No whitespace</div>
            </div>
            """,
            unsafe_allow_html=True
        )

def render_driver_profile_assignment():
    render_browser_header("drivers/profile")
    
    st.markdown("## Driver Profile & Vehicle Assignment")
    
    driver = st.session_state.current_driver
    
    c_photo, c_info = st.columns([1, 4])
    with c_photo:
        st.markdown("<div style='width:100px; height:100px; border-radius:50%; background:#CBD5E1; display:flex; align-items:center; justify-content:center; font-size:3rem;'>👤</div>", unsafe_allow_html=True)
        st.file_uploader("Profile photo (optional)", type=["png", "jpg"])
        
    with c_info:
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("First Name", value="Rahul")
            st.text_input("Date of Birth", value="14 / 08 / 1994")
            st.text_input("Shift Type", value="Night (10 PM – 6 AM)")
            st.text_input("Emergency Contact", value="9876500011 (optional)")
        with c2:
            st.text_input("Last Name", value="Verma")
            st.text_input("Licence Number", value="UP32 20190012345")
            st.text_input("Assigned Vehicle ID", value="UP32-AB-1234")
            st.text_input("Calibration Status", value="Completed", disabled=True)
            
        if st.button("Save Profile & Assignment", type="primary"):
            st.success("Profile details updated successfully.")

def render_driver_scorecards():
    render_browser_header("drivers")
    
    st.markdown("## Driver Scorecards")
    
    c_list, c_details = st.columns([2, 3])
    
    with c_list:
        st.text_input("🔍 Search driver name / ID")
        drivers = st.session_state.drivers_list
        for d in drivers:
            dot_color = "#EF4444" if d['status_color'] == 'red' else "#F59E0B" if d['status_color'] == 'amber' else "#10B981"
            st.markdown(
                f"""
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:12px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <strong style="color:#0F172A;">{d['name']}</strong>
                        <div style="font-size:0.8rem; color:#64748B;">{d['incidents']} incidents</div>
                    </div>
                    <div style="width:14px; height:14px; border-radius:50%; background:{dot_color};"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with c_details:
        st.markdown("### Rahul Verma • DRV2026A")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trips", "42")
        m2.metric("Incidents", "7")
        m3.metric("Avg score", "0.41")
        m4.metric("High sev.", "2")
        
        st.markdown("#### 30-day fatigue trend")
        x_trend = np.linspace(1, 30, 10)
        y_trend = [0.2, 0.3, 0.25, 0.4, 0.5, 0.35, 0.6, 0.45, 0.7, 0.62]
        fig_tr = px.line(x=x_trend, y=y_trend)
        fig_tr.update_traces(line_color='#1E40AF', line_width=3)
        fig_tr.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_tr, use_container_width=True)

        st.markdown("#### Severity breakdown")
        st.markdown("""
        <div style="margin-bottom:6px;">Low &nbsp;<span style="background:#10B981; padding:2px 120px; border-radius:4px; color:white; font-weight:700;">3</span></div>
        <div style="margin-bottom:6px;">Medium &nbsp;<span style="background:#F59E0B; padding:2px 80px; border-radius:4px; color:white; font-weight:700;">2</span></div>
        <div>High &nbsp;<span style="background:#EF4444; padding:2px 80px; border-radius:4px; color:white; font-weight:700;">2</span></div>
        """, unsafe_allow_html=True)

def render_incident_review():
    render_browser_header("incidents")
    
    st.markdown("## Incident Review Workflow")
    
    f1, f2, f3, f4 = st.columns(4)
    f1.selectbox("Vehicle", ["All Vehicles", "UP32-AB-1234", "UP32-CD-5678"])
    f2.selectbox("Driver", ["All Drivers", "Rahul Verma", "Amit Singh"])
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
