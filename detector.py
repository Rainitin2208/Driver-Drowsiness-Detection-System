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

        # MediaPipe initialization (tries Tasks API and legacy FaceMesh; records why it failed)
        self.use_tasks = False
        self.face_mesh = None
        self.landmarker = None
        self.engine_error = None
        self._infer_lock = threading.Lock()

        errors = []
        if USE_TASKS_API or mp_solutions_facemesh is None:
            self._init_tasks_landmarker(errors)
        if self.landmarker is None and mp_solutions_facemesh is not None:
            self._init_legacy_face_mesh(errors)
        if self.landmarker is None and self.face_mesh is None:
            self.engine_error = " | ".join(errors) if errors else "MediaPipe engine unavailable"
            print(f"[DrowsiGuard] MediaPipe failed to load: {self.engine_error}", flush=True)

        # OpenCV Haar Cascade fallback detectors
        self.face_cascade = None
        self.eye_cascade = None
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        except Exception:
            self.face_cascade = None
            self.eye_cascade = None

    def _init_tasks_landmarker(self, errors):
        try:
            from mediapipe.tasks.python import vision, BaseOptions
            model_file = ensure_mediapipe_model_exists()
            try:
                base = BaseOptions(model_asset_path=str(model_file), delegate=BaseOptions.Delegate.CPU)
            except Exception:
                base = BaseOptions(model_asset_path=str(model_file))
            options = vision.FaceLandmarkerOptions(
                base_options=base,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            self.use_tasks = True
        except Exception as e:
            self.landmarker = None
            self.use_tasks = False
            errors.append(f"tasks: {type(e).__name__}: {e}")

    def _init_legacy_face_mesh(self, errors):
        try:
            self.face_mesh = mp_solutions_facemesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            self.face_mesh = None
            errors.append(f"solutions: {type(e).__name__}: {e}")

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
        frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        
        try:
            self._infer_lock.acquire()
            if self.use_tasks and self.landmarker is not None:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                res = self.landmarker.detect(mp_image)
                if res and res.face_landmarks and len(res.face_landmarks) > 0:
                    landmarks = res.face_landmarks[0]
                else:
                    # Fallback: input frame might already be in RGB format
                    frame_direct = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
                    mp_image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_direct)
                    res2 = self.landmarker.detect(mp_image2)
                    if res2 and res2.face_landmarks and len(res2.face_landmarks) > 0:
                        landmarks = res2.face_landmarks[0]
            elif self.face_mesh is not None:
                results = self.face_mesh.process(frame_rgb)
                if results and results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                    landmarks = results.multi_face_landmarks[0]
                else:
                    results2 = self.face_mesh.process(frame_bgr)
                    if results2 and results2.multi_face_landmarks and len(results2.multi_face_landmarks) > 0:
                        landmarks = results2.multi_face_landmarks[0]
        except Exception:
            landmarks = None
        finally:
            try:
                self._infer_lock.release()
            except RuntimeError:
                pass

        # Stage 3 Fallback: Haar Cascade + MediaPipe cropped detection
        haar_face_box = None
        if landmarks is None and self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(60, 60))
                if len(faces) > 0:
                    fx, fy, fw, fh = [int(v) for v in faces[0]]
                    haar_face_box = (fx, fy, fw, fh)
                    
                    pad_x, pad_y = int(fw * 0.15), int(fh * 0.15)
                    cx1, cy1 = max(0, fx - pad_x), max(0, fy - pad_y)
                    cx2, cy2 = min(img_w, fx + fw + pad_x), min(img_h, fy + fh + pad_y)
                    face_crop = frame_bgr[cy1:cy2, cx1:cx2]
                    
                    if face_crop.size > 0 and self.use_tasks and self.landmarker is not None:
                        crop_rgb = np.ascontiguousarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), dtype=np.uint8)
                        mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                        crop_res = self.landmarker.detect(mp_crop)
                        if crop_res and crop_res.face_landmarks and len(crop_res.face_landmarks) > 0:
                            crop_lms = crop_res.face_landmarks[0]
                            cw, ch = (cx2 - cx1), (cy2 - cy1)
                            for lm in crop_lms:
                                lm.x = (cx1 + lm.x * cw) / img_w
                                lm.y = (cy1 + lm.y * ch) / img_h
                            landmarks = crop_lms
            except Exception:
                pass

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
                        if self.is_drowsy:
                            status_str = f"DROWSINESS DETECTED! (EAR: {self.current_ear:.2f})"
                            status_color = (0, 0, 255)
                        elif self.is_yawning:
                            status_str = f"Face Found | YAWN DETECTED! (MAR: {self.current_mar:.2f})"
                            status_color = (0, 165, 255)
                        else:
                            status_str = f"Face Found | EAR: {self.current_ear:.2f}"
                            status_color = (16, 185, 129)
                        text_y = max(25, min_y - 10)
                        cv2.putText(annotated_frame, status_str, (min_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                    
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
            elif haar_face_box is not None:
                self.face_detected = True
                fx, fy, fw, fh = haar_face_box
                
                # Check eyes in upper face region
                eyes = []
                if self.eye_cascade is not None:
                    try:
                        gray_face = cv2.cvtColor(frame_bgr[fy:fy+fh, fx:fx+fw], cv2.COLOR_BGR2GRAY)
                        upper_face = gray_face[:int(fh * 0.6), :]
                        eyes = self.eye_cascade.detectMultiScale(upper_face, scaleFactor=1.1, minNeighbors=3)
                    except Exception:
                        eyes = []
                
                if len(eyes) >= 1:
                    self.current_ear = 0.28
                    self.drowsy_counter = 0
                    if self.is_drowsy:
                        self.is_drowsy = False
                        self.alert_stage = 0
                        stop_sound_alarm_pygame()
                else:
                    self.drowsy_counter += 1
                    if self.drowsy_counter >= 4:
                        self.current_ear = 0.18
                        if self.drowsy_counter >= self.consec_frames:
                            if not self.is_drowsy:
                                self.is_drowsy = True
                                self.total_alerts += 1
                                self.alert_stage = 1
                            if self.alarm_enabled and not self.muted_chime:
                                play_sound_alarm_pygame()
                
                box_color = (0, 0, 255) if self.is_drowsy else (16, 185, 129)
                cv2.rectangle(annotated_frame, (fx, fy), (fx + fw, fy + fh), box_color, 2)
                status_str = f"DROWSINESS DETECTED! (EAR: {self.current_ear:.2f})" if self.is_drowsy else f"Face Found | EAR: {self.current_ear:.2f}"
                cv2.putText(annotated_frame, status_str, (fx, max(25, fy - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                
                elapsed = round(time.time() - self.session_start_time, 1)
                self.ear_history.append((elapsed, self.current_ear))
                self.fused_score = self.compute_fused_score()
                self.fatigue_history.append((elapsed, self.fused_score))
            else:
                self.face_detected = False
                cv2.putText(annotated_frame, "Searching for face...", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
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
