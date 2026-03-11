# 🤟 Sign Language Detector

> Real-time American Sign Language (ASL) gesture recognition using MediaPipe and LSTM neural networks — trained on custom self-collected data.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-85%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📌 Overview

This project implements a **real-time sign language detection system** that recognizes dynamic hand gestures using a webcam. Unlike static image classifiers, this system captures **temporal sequences** of hand landmarks using MediaPipe and feeds them into an **LSTM (Long Short-Term Memory)** neural network — enabling recognition of gestures that unfold over time.

The model was trained entirely on **self-collected data**, making it robust to personal hand shapes and lighting conditions.

---

## 🎯 Supported Gestures

| Gesture | Label |
|---|---|
| 👋 Hello | `hello` |
| ✅ Yes | `yes` |
| ❌ No | `no` |
| 🙏 Thank You | `thanku` |
| 🤟 I Love You | `iloveyou` |

---

## 🧠 How It Works

```
Webcam Feed
    ↓
MediaPipe Holistic
(Hand + Pose + Face Landmarks — 1662 keypoints per frame)
    ↓
Sequence Collector
(30 frames per gesture = 1 temporal sequence)
    ↓
LSTM Neural Network
(3 stacked LSTM layers → Dense → Softmax)
    ↓
Real-time Gesture Prediction
(Label + Confidence Score overlay on webcam)
```

### Why LSTM and not CNN?

Sign language gestures are **dynamic** — they unfold across time. A single frame of "hello" looks almost identical to a static hand wave. LSTM networks are designed to learn **patterns across time sequences**, making them ideal for gesture recognition. A standard CNN would only see one frame at a time and miss the motion entirely.

### Why MediaPipe?

MediaPipe Holistic extracts **1662 keypoint coordinates** per frame — capturing precise positions of hands, face, and body pose. This means the model learns from **skeletal structure**, not pixel color, making it invariant to skin tone, lighting, and background. This is the same approach used in production systems at Google.

---

## 📁 Project Structure

```
sign-language-detector/
│
├── collect_sequences.py      # Webcam-based data collection pipeline
├── process_sequences.py      # Keypoint extraction and sequence processing
├── train_lstm_model.py       # LSTM model architecture and training
├── inference_dynamic.py      # Real-time inference with live webcam feed
├── screenshots/              # Demo screenshots
└── .gitignore
```

---

## ⚙️ Technical Stack

| Component | Technology |
|---|---|
| Landmark Extraction | MediaPipe Holistic |
| Model Architecture | LSTM (TensorFlow/Keras) |
| Data Collection | OpenCV + Custom Pipeline |
| Real-time Inference | OpenCV + NumPy |
| Language | Python 3.8+ |

---

## 🏗️ Model Architecture

```
Input: (30 frames × 1662 keypoints)
    ↓
LSTM Layer 1 — 64 units, return_sequences=True
    ↓
LSTM Layer 2 — 128 units, return_sequences=True
    ↓
LSTM Layer 3 — 64 units
    ↓
Dense Layer — 64 units, ReLU activation
    ↓
Dense Layer — 32 units, ReLU activation
    ↓
Output Layer — 5 units, Softmax activation
    ↓
Prediction: Gesture Label + Confidence Score
```

**Training Accuracy: ~85%** on self-collected dataset

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow mediapipe opencv-python numpy matplotlib
```

### Step 1 — Collect Your Own Data
```bash
python collect_sequences.py
```
This opens your webcam and records 30-frame sequences for each gesture. Collect at least 30 sequences per gesture for good accuracy.

### Step 2 — Process the Sequences
```bash
python process_sequences.py
```
Extracts MediaPipe keypoints from each recorded sequence and saves them as NumPy arrays.

### Step 3 — Train the Model
```bash
python train_lstm_model.py
```
Trains the LSTM network on your processed sequences. Training takes ~5–10 minutes on CPU.

### Step 4 — Run Live Inference
```bash
python inference_dynamic.py
```
Opens webcam feed with real-time gesture detection overlay.

---

## 📊 Results

- **Training Accuracy:** ~85%
- **Inference Speed:** Real-time (~30 FPS on CPU)
- **Gestures Supported:** 5 dynamic ASL signs
- **Data:** 100% self-collected via webcam

---

## 🔬 Key Technical Concepts

**Temporal Sequences:** Each gesture is represented as 30 consecutive frames, capturing motion over ~1 second of real-time video.

**Keypoint Normalization:** MediaPipe landmarks are normalized relative to the frame dimensions, making the model resolution-independent.

**Confidence Thresholding:** Predictions below a confidence threshold are suppressed to reduce false positives during inference.

---

## 🛣️ Future Improvements

- [ ] Expand vocabulary to 20+ signs
- [ ] Add sentence formation (sequence of gestures → full sentence)
- [ ] Deploy as a web app using FastAPI + WebRTC for browser-based inference
- [ ] Integrate text-to-speech for accessibility output
- [ ] Train on larger dataset for improved generalization across users

---

## 🎓 Research References

- Zhang, F. et al. (2020). *MediaPipe Hands: On-device Real-time Hand Tracking.* Google Research.
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8).
- Lugaresi, C. et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines.* Google Research.

---

## 👨‍💻 Author

**Aldrin Johny**
B.Tech Computer Science (AI), ASIET Kalady, Kerala
[GitHub](https://github.com/aldrinjohny-ai)

---

## 📄 License

This project is licensed under the MIT License.

---

> *"Built to bridge communication gaps between the deaf community and the hearing world through accessible AI."*
