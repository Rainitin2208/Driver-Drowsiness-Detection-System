---
title: Driver Drowsiness Detection System
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
---

# 🚗 AI-Based Driver Drowsiness Detection System

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-orange.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

A **production-ready, real-time AI Driver Drowsiness Detection System** built with **Python, OpenCV, MediaPipe Face Mesh, Streamlit, and `streamlit-webrtc`**. 

This system continuously tracks face mesh landmarks from a live webcam feed, computes the Eye Aspect Ratio (EAR) for both eyes, detects prolonged eye closure, tracks driver blinks, triggers audible multi-frequency warning alarms, and visualizes real-time telemetry analytics on an interactive glassmorphism dashboard.

---

## 🌟 Key Features

* 🎥 **Real-time Webcam Stream via WebRTC**: Seamless browser webcam integration powered by `streamlit-webrtc`, supporting local running and cloud deployment (Streamlit Cloud & Hugging Face Spaces).
* 👁️ **High-Precision MediaPipe Face Mesh Tracking**: Detects 468 3D facial landmarks and isolates 6 key points per eye for scale-invariant EAR calculations.
* ⚡ **Real-Time Eye Aspect Ratio (EAR)**: Dynamic math calculation comparing vertical and horizontal eye dimensions.
* 🚨 **Multi-Modal Alarm System**: Plays loud host machine audio via Pygame AND streams base64 HTML5 audio directly into remote web browsers when drowsiness is detected.
* 📊 **Interactive Telemetry & Plotly Analytics**: Real-time streaming graphs for EAR vs Time, Gauge meters for eye openness, blink counters, FPS monitor, and alert counters.
* 📁 **Event Logging & CSV Export**: Automatic logging of drowsiness duration, timestamps, and EAR values into `data/history.csv` with a 1-click Download button.
* 📸 **Snapshot Capture**: Instant frame capture saving annotated snapshots into `screenshots/`.
* 🎨 **Modern Glassmorphism UI**: Dark/Light mode toggle, dynamic status badges, custom styling, responsive metric grid.

---

## 🛠️ Tech Stack

* **Language**: Python 3.11+
* **Computer Vision**: OpenCV, MediaPipe Face Mesh (`mediapipe`)
* **Web Framework**: Streamlit, `streamlit-webrtc`
* **Video Encoding**: PyAV (`av`)
* **Audio**: Pygame Mixer, HTML5 Web Audio API
* **Data Visualization**: Plotly, Pandas
* **Image Processing**: Pillow (PIL), NumPy

---

## 📂 Project Structure

```
driver_drowsiness_detection/
├── app.py                # Main Streamlit Dashboard UI & WebRTC Integration
├── detector.py           # DrowsinessDetector & Thread-Safe VideoProcessor Engine
├── ear.py                # EAR Calculation Math & Landmark Extraction
├── config.py             # Global Constants, Landmark Mapping & Styling Tokens
├── utils.py              # Sound Generator, Branding Assets, CSV Logger & Screenshots
├── requirements.txt      # Python Dependencies with Version Pins
├── .gitignore            # Git Exclusions
├── README.md             # Documentation & Guide
├── alarm.wav             # Auto-generated Warning Alarm Sound
├── assets/
│   ├── logo.png          # System Branding Logo
│   └── banner.png        # System Banner Image
├── data/
│   └── history.csv       # Drowsiness Detection Event Logs
└── screenshots/          # Saved Frame Snapshots
```

---

## 🚀 Quick Start & Local Setup

### 1. Clone or Download Repository
```bash
git clone https://github.com/your-username/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Application
```bash
streamlit run app.py
```
Open your web browser at `http://localhost:8501`.

---

## 🧠 EAR Algorithm Math

The Eye Aspect Ratio (EAR) is calculated using 6 key landmark points per eye:

$$\text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$

Where:
* $p_1, p_4$ are horizontal eye corner points.
* $p_2, p_3, p_5, p_6$ are top and bottom eyelid points.

When the eye is open, EAR remains around **0.28 - 0.35**. When eyes close, EAR drops close to **0.0 - 0.15**. If EAR stays below the configured threshold (default `0.25`) for `16` consecutive frames, the system triggers the drowsiness alarm.

---

## ☁️ Deployment Guide

### Deploying on Streamlit Community Cloud
1. Push this project repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**, select your repository, branch `main`, and main file `app.py`.
4. Deploy! `streamlit-webrtc` automatically routes webcam streams via public STUN servers.

### Deploying on Hugging Face Spaces
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) and select **Streamlit** SDK.
2. Push all project files to your Space repository.
3. Hugging Face Spaces will automatically build `requirements.txt` and launch `app.py`.

---

## 🖼️ Screenshots

| Live WebRTC Feed | Real-time Analytics |
| :---: | :---: |
| *Annotated webcam feed with MediaPipe mesh & EAR HUD* | *Streaming Plotly EAR vs Time graph & Gauge meter* |

---

## 🔮 Future Scope

* 😴 **Yawn Detection**: Incorporate Mouth Aspect Ratio (MAR) to detect yawning frequency.
* 👤 **Head Pose Estimation**: Track pitch, yaw, and roll to alert when driver tilts head down or loses focus on the road.
* 🌙 **IR / Low-Light Support**: Optimize detection pipeline for night driving IR cameras.
* 📲 **Mobile App / Edge Device Deployment**: Port model inference to Raspberry Pi 4 / Jetson Nano for embedded car integration.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
