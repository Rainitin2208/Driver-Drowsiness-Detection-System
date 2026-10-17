"""
Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and Head Pose Estimation Module.
Provides Euclidean distance utility and MediaPipe face mesh landmark extraction.
"""

import math
import cv2
import numpy as np

def euclidean_distance(point1, point2):
    """
    Compute 2D Euclidean distance between two points (x, y).
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR) given 6 eye landmark coordinates.
    Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)
    """
    if len(eye_landmarks) < 6:
        return 0.0
    
    p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]
    
    v1 = euclidean_distance(p2, p6)
    v2 = euclidean_distance(p3, p5)
    h = euclidean_distance(p1, p4)
    
    if h == 0.0:
        return 0.0
    
    ear = (v1 + v2) / (2.0 * h)
    return round(float(ear), 4)

def calculate_mar(mouth_landmarks):
    """
    Calculate Mouth Aspect Ratio (MAR) for Yawning detection.
    Formula: MAR = ||p_top - p_bottom|| / ||p_left - p_right||
    """
    if len(mouth_landmarks) < 4:
        return 0.0
    
    p_top, p_bottom, p_left, p_right = mouth_landmarks[:4]
    
    v = euclidean_distance(p_top, p_bottom)
    h = euclidean_distance(p_left, p_right)
    
    if h == 0.0:
        return 0.0
    
    mar = v / h
    return round(float(mar), 4)

def extract_landmark_coords(face_landmarks, landmark_indices, img_w, img_h):
    """
    Extract pixel coordinates for specific face mesh landmark indices.
    """
    coords = []
    landmarks_seq = face_landmarks.landmark if hasattr(face_landmarks, 'landmark') else face_landmarks
    
    for idx in landmark_indices:
        if idx < len(landmarks_seq):
            landmark = landmarks_seq[idx]
            x = int(landmark.x * img_w)
            y = int(landmark.y * img_h)
            coords.append((x, y))
    return coords

# Legacy alias
extract_eye_coords = extract_landmark_coords

def estimate_head_pose(face_landmarks, img_w, img_h):
    """
    Estimate Head Pose Angles (Pitch, Yaw, Roll) in degrees using cv2.solvePnP.
    """
    landmarks_seq = face_landmarks.landmark if hasattr(face_landmarks, 'landmark') else face_landmarks
    if len(landmarks_seq) < 468:
        return 0.0, 0.0, 0.0
    
    # 2D image points from landmarks: Nose(1), Chin(152), Left Eye(33), Right Eye(263), Left Mouth(61), Right Mouth(291)
    indices = [1, 152, 33, 263, 61, 291]
    image_points = []
    for idx in indices:
        lm = landmarks_seq[idx]
        image_points.append([lm.x * img_w, lm.y * img_h])
    image_points = np.array(image_points, dtype=np.float64)
    
    # Generic 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)
    
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return 0.0, 0.0, 0.0
    
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    
    # Euler angles
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch, yaw, roll = euler_angles[0][0], euler_angles[1][0], euler_angles[2][0]
    
    return round(float(pitch), 1), round(float(yaw), 1), round(float(roll), 1)
