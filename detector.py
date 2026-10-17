"""
Drowsiness Detector Engine (DrowsiGuard).
Tracks Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR for Yawning), Head Pose Angles (Pitch/Yaw/Roll),
Wearable Heart Rate / HRV, Fused Fatigue Score, and Escalating Alert Stages.
Includes Streamlit-WebRTC VideoProcessorBase implementation for smooth real-time streaming.
"""

import time
import threading
import os
from collections import deque
import cv2
import numpy as np
import mediapipe as mp

from config import (
    RIGHT_EYE_LANDMARKS, LEFT_EYE_LANDMARKS, MOUTH_LANDMARKS,
    DEFAULT_EAR_THRESHOLD, DEFAULT_CONSEC_FRAMES,
    DEFAULT_YAWN_MAR_THRESHOLD, DEFAULT_YAWN_LIMIT,
    DEFAULT_HEAD_TILT_ANGLE, DEFAULT_HEAD_TILT_DURATION,
    DEFAULT_WARNING_FATIGUE_SCORE, DEFAULT_CRITICAL_FATIGUE_SCORE,
    MAX_CHART_HISTORY, COLORS, MODEL_PATH
)
from ear import calculate_ear, calculate_mar, extract_landmark_coords, estimate_head_pose
from utils import play_sound_alarm_pygame, stop_sound_alarm_pygame, EventLogger, ensure_mediapipe_model_exists

# WebRTC VideoProcessor import
try:
    from streamlit_webrtc import VideoProcessorBase
    import av
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False
    class VideoProcessorBase:
        pass

# MediaPipe detection engine configuration
USE_TASKS_API = False
mp_solutions_facemesh = None

try:
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        mp_solutions_facemesh = mp.solutions.face_mesh
    else:
        import mediapipe.solutions.face_mesh as mp_solutions_facemesh
except Exception:
    mp_solutions_facemesh = None

if mp_solutions_facemesh is None:
    try:
        from mediapipe.tasks.python import vision, BaseOptions
        USE_TASKS_API = True
    except Exception:
        USE_TASKS_API = False


class DrowsinessDetector:
    """
    State management and computer-vision processing engine for DrowsiGuard.
    """
    def __init__(self, ear_threshold=DEFAULT_EAR_THRESHOLD, consec_frames=DEFAULT_CONSEC_FRAMES, alarm_enabled=True):
        self.lock = threading.Lock()
        
        # Configurations & Thresholds
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.yawn_mar_threshold = DEFAULT_YAWN_MAR_THRESHOLD
        self.head_tilt_threshold = DEFAULT_HEAD_TILT_ANGLE
        self.alarm_enabled = alarm_enabled
        
        # Vision metrics
        self.current_ear = 0.30
        self.ear_left = 0.30
        self.ear_right = 0.30
        self.current_mar = 0.15
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        
        # Sensor & Fatigue metrics
        self.wearable_connected = True
        self.heart_rate = 72
        self.hrv = 48
        self.steering_variance = 0.50
        
        # Event counters
        self.drowsy_counter = 0
        self.blink_count = 16
        self.yawn_count = 1
        self.total_alerts = 0
        self.is_drowsy = False
        self.is_yawning = False
        self.face_detected = True
        
        # Trip state
        self.trip_active = True
        self.session_start_time = time.time()
        self.drowsy_event_start = None
        self.unacknowledged_start = None
        
        # Alert stage (0: Normal, 1: Soft Chime, 2: Buzzer + Vibration, 3: High Sev + Hazard)
        self.alert_stage = 0
        self.muted_chime = False
        
        # Temporal history for live charts
        self.ear_history = deque(maxlen=MAX_CHART_HISTORY)
        self.mar_history = deque(maxlen=MAX_CHART_HISTORY)
        self.pitch_history = deque(maxlen=MAX_CHART_HISTORY)
        self.fatigue_history = deque(maxlen=MAX_CHART_HISTORY)
        
        # Fused Fatigue Score (0.0 to 1.0)
        self.fused_score = 0.22
        
        # Logger
        self.logger = EventLogger()
        self._was_eye_closed = False

        # MediaPipe initialization
        self.use_tasks = USE_TASKS_API
        self.face_mesh = None
        self.landmarker = None

        if self.use_tasks:
            try:
                model_file = ensure_mediapipe_model_exists()
                from mediapipe.tasks.python import vision, BaseOptions
                options = vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_file)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
            except Exception:
                self.landmarker = None
        elif mp_solutions_facemesh is not None:
            try:
                self.face_mesh = mp_solutions_facemesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception:
                self.face_mesh = None

    def update_settings(self, ear_threshold, consec_frames, alarm_enabled):
        with self.lock:
            self.ear_threshold = ear_threshold
            self.consec_frames = consec_frames
            self.alarm_enabled = alarm_enabled

    def compute_fused_score(self):
        ear_score = max(0.0, min(1.0, (0.32 - self.current_ear) / 0.15)) if self.current_ear < 0.32 else 0.0
        yawn_score = min(1.0, self.yawn_count / 4.0)
        head_score = min(1.0, abs(self.pitch) / 30.0)
        hr_score = max(0.0, min(1.0, (75 - self.heart_rate) / 25.0)) if self.wearable_connected else 0.0
        steering_score = self.steering_variance
        
        if self.wearable_connected:
            fused = (0.35 * ear_score) + (0.15 * yawn_score) + (0.15 * head_score) + (0.20 * hr_score) + (0.15 * steering_score)
        else:
            fused = (0.45 * ear_score) + (0.20 * yawn_score) + (0.20 * head_score) + (0.15 * steering_score)
            
        return round(float(max(0.0, min(1.0, fused))), 2)

    def acknowledge_alert(self):
        with self.lock:
            self.alert_stage = 0
            self.is_drowsy = False
            self.unacknowledged_start = None
            stop_sound_alarm_pygame()

    def process_frame(self, frame_bgr):
        img_h, img_w, _ = frame_bgr.shape
        annotated_frame = frame_bgr.copy()
        
        landmarks = None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        try:
            if self.use_tasks and self.landmarker is not None:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                res = self.landmarker.detect(mp_image)
                if res and res.face_landmarks and len(res.face_landmarks) > 0:
                    landmarks = res.face_landmarks[0]
            elif self.face_mesh is not None:
                results = self.face_mesh.process(frame_rgb)
                if results and results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                    landmarks = results.multi_face_landmarks[0]
        except Exception:
            landmarks = None

        with self.lock:
            if landmarks is not None:
                self.face_detected = True
                
                left_eye_pts = extract_landmark_coords(landmarks, LEFT_EYE_LANDMARKS, img_w, img_h)
                right_eye_pts = extract_landmark_coords(landmarks, RIGHT_EYE_LANDMARKS, img_w, img_h)
                mouth_pts = extract_landmark_coords(landmarks, MOUTH_LANDMARKS, img_w, img_h)
                
                if len(left_eye_pts) >= 6 and len(right_eye_pts) >= 6:
                    self.ear_left = calculate_ear(left_eye_pts)
                    self.ear_right = calculate_ear(right_eye_pts)
                    self.current_ear = round((self.ear_left + self.ear_right) / 2.0, 4)
                    
                    if len(mouth_pts) >= 4:
                        self.current_mar = calculate_mar(mouth_pts)
                    
                    self.pitch, self.yaw, self.roll = estimate_head_pose(landmarks, img_w, img_h)
                    
                    elapsed = round(time.time() - self.session_start_time, 1)
                    self.ear_history.append((elapsed, self.current_ear))
                    self.mar_history.append((elapsed, self.current_mar))
                    self.pitch_history.append((elapsed, self.pitch))
                    
                    if self.current_mar > self.yawn_mar_threshold:
                        if not self.is_yawning:
                            self.is_yawning = True
                            self.yawn_count += 1
                    else:
                        self.is_yawning = False
                    
                    eye_color = (0, 0, 255) if self.is_drowsy else (0, 255, 0)
                    left_hull = cv2.convexHull(np.array(left_eye_pts))
                    right_hull = cv2.convexHull(np.array(right_eye_pts))
                    cv2.polylines(annotated_frame, [left_hull], True, eye_color, 2)
                    cv2.polylines(annotated_frame, [right_hull], True, eye_color, 2)
                    
                    # Mouth contour & Yawn overlay
                    if len(mouth_pts) >= 4:
                        mouth_hull = cv2.convexHull(np.array(mouth_pts))
                        mouth_color = (0, 165, 255) if self.is_yawning else (0, 255, 255)
                        cv2.polylines(annotated_frame, [mouth_hull], True, mouth_color, 2)
                    
                    all_pts = left_eye_pts + right_eye_pts + mouth_pts
                    if all_pts:
                        xs = [p[0] for p in all_pts]
                        ys = [p[1] for p in all_pts]
                        min_x, max_x = max(10, min(xs) - 40), min(img_w - 10, max(xs) + 40)
                        min_y, max_y = max(10, min(ys) - 60), min(img_h - 10, max(ys) + 60)
                        box_color = (0, 0, 255) if self.is_drowsy else ((0, 165, 255) if self.is_yawning else (16, 185, 129))
                        cv2.rectangle(annotated_frame, (min_x, min_y), (max_x, max_y), box_color, 2)

                    
                    if self.current_ear < self.ear_threshold:
                        self.drowsy_counter += 1
                        if self.drowsy_counter >= self.consec_frames:
                            if not self.is_drowsy:
                                self.is_drowsy = True
                                self.total_alerts += 1
                                self.drowsy_event_start = time.time()
                                self.unacknowledged_start = time.time()
                                self.alert_stage = 1
                            
                            if self.unacknowledged_start and (time.time() - self.unacknowledged_start > 10):
                                self.alert_stage = 2
                            if self.unacknowledged_start and (time.time() - self.unacknowledged_start > 25):
                                self.alert_stage = 3
                                
                            if self.alarm_enabled and not self.muted_chime:
                                play_sound_alarm_pygame()
                    else:
                        # Eyes are OPEN
                        if 2 <= self.drowsy_counter < self.consec_frames:
                            # Quick eye closure that reopened before threshold = Normal Blink!
                            self.blink_count += 1
                        
                        # Reset counter on eye reopen so normal blinks don't accumulate
                        self.drowsy_counter = 0
                        
                        if self.is_drowsy and self.alert_stage <= 1:
                            duration = time.time() - self.drowsy_event_start if self.drowsy_event_start else 0.0
                            self.logger.log_event(self.current_ear, "DROWSINESS DETECTED", duration)
                            self.is_drowsy = False
                            self.alert_stage = 0
                            stop_sound_alarm_pygame()

                
                self.fused_score = self.compute_fused_score()
                self.fatigue_history.append((elapsed, self.fused_score))

                # Render Real-Time HUD on Frame
                # 1. Top Status Banner
                if self.is_drowsy:
                    cv2.rectangle(annotated_frame, (0, 0), (img_w, 42), (0, 0, 220), -1)
                    cv2.putText(annotated_frame, f"*** ALERT: DROWSINESS DETECTED! WAKE UP! (EAR: {self.current_ear:.2f}) ***",
                                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
                elif self.is_yawning:
                    cv2.rectangle(annotated_frame, (0, 0), (img_w, 42), (0, 140, 255), -1)
                    cv2.putText(annotated_frame, f"WARNING: YAWNING DETECTED (MAR: {self.current_mar:.2f})",
                                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
                else:
                    cv2.rectangle(annotated_frame, (0, 0), (img_w, 36), (15, 120, 20), -1)
                    cv2.putText(annotated_frame, f"DRIVER AWAKE & ATTENTIVE | EAR: {self.current_ear:.2f}",
                                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

                # 2. Bottom-Left Live Telemetry Overlay Box
                box_h = 105
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (10, img_h - box_h - 10), (340, img_h - 10), (10, 15, 25), -1)
                cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                cv2.rectangle(annotated_frame, (10, img_h - box_h - 10), (340, img_h - 10), (0, 180, 216), 1)

                eye_state = "CLOSED (ALERT!)" if self.current_ear < self.ear_threshold else "OPEN"
                eye_color = (0, 0, 255) if self.current_ear < self.ear_threshold else (0, 255, 0)
                cv2.putText(annotated_frame, f"EAR: {self.current_ear:.3f} (Limit: {self.ear_threshold:.2f})",
                            (20, img_h - box_h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                cv2.putText(annotated_frame, f"Eyes: {eye_state}",
                            (20, img_h - box_h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, eye_color, 1)
                cv2.putText(annotated_frame, f"MAR: {self.current_mar:.3f} | Yawns: {self.yawn_count}",
                            (20, img_h - box_h + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                cv2.putText(annotated_frame, f"Head Pose: Pitch {self.pitch:+.1f} deg | Yaw {self.yaw:+.1f} deg",
                            (20, img_h - box_h + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                cv2.putText(annotated_frame, f"Fatigue Risk: {int(self.fused_score * 100)}% | Blinks: {self.blink_count}",
                            (20, img_h - box_h + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
            else:
                self.face_detected = False
                # Visual guide when face is searching
                cv2.rectangle(annotated_frame, (0, 0), (img_w, 40), (0, 140, 255), -1)
                cv2.putText(annotated_frame, "SEARCHING FOR FACE... PLEASE LOOK TOWARDS CAMERA",
                            (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
                center_x, center_y = img_w // 2, img_h // 2
                cv2.ellipse(annotated_frame, (center_x, center_y), (int(img_w * 0.22), int(img_h * 0.35)), 0, 0, 360, (0, 180, 216), 2)
                cv2.putText(annotated_frame, "Align Face Here", (center_x - 70, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 216), 2)
                
        return annotated_frame


class DrowsinessVideoProcessor(VideoProcessorBase):
    """
    Streamlit-WebRTC Video Processor Class.
    Implements continuous frame processing loop via recv(frame).
    """
    detector_ref = None

    def __init__(self):
        self.detector = DrowsinessVideoProcessor.detector_ref
        if self.detector is None:
            self.detector = DrowsinessDetector()

    def recv(self, frame):
        if not HAS_WEBRTC:
            return frame
        img = frame.to_ndarray(format="bgr24")
        if self.detector is not None:
            annotated = self.detector.process_frame(img)
        else:
            annotated = img
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
