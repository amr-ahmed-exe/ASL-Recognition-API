<div align="center">

#  ASL Recognition API

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-GCN-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![WebSocket](https://img.shields.io/badge/WebSocket-Real--Time-010101?style=for-the-badge&logo=socket.io&logoColor=white)

**Real-time American Sign Language recognition API powered by Graph Convolutional Networks and MediaPipe hand landmarks.**

*Graduation Project · Suez Canal University · Class of 2026*

[Demo](#demo) · [Architecture](#architecture) · [API Reference](#api-reference) · [Getting Started](#getting-started)

</div>

---

##  Overview

**Sign AI** is a cloud-based ASL recognition system that translates hand gestures into text in real-time. The Flutter mobile client captures hand landmarks using MediaPipe and streams them to this FastAPI backend via WebSocket. The server runs inference through a PyTorch **Graph Convolutional Network (GCN)**, returning letter predictions with confidence scores, spell-check suggestions, and auto-spacing.

###  Key Highlights

-  **84.2% perceptual sign recognition accuracy** across all 26 ASL letters
-  **Real-time inference** via WebSocket — sub-100ms response per frame
-  **Zero error rate** under 100 concurrent WebSocket connections
-  **Graph Convolutional Network** trained on 21 MediaPipe hand landmarks
-  **Dynamic gesture support** for motion-based letters J and Z
-  **Smart typing engine** with debouncing, auto-space, and spell-check

---

##  Architecture

```
Flutter Mobile App (MediaPipe)
         │
         │  WebSocket /ws/predict
         │  [21 hand landmark points per frame]
         ▼
┌─────────────────────────────────────┐
│         ASL Recognition API         │
│                                      │
│  ┌─────────────────────────────┐    │
│  │   LandmarkFilter (EMA)      │    │  ← Smooths jitter from raw input
│  └────────────┬────────────────┘    │
│               │                      │
│  ┌────────────▼────────────────┐    │
│  │  predict_from_pytorch()     │    │  ← GCN inference + J/Z motion
│  └────────────┬────────────────┘    │
│               │                      │
│  ┌────────────▼────────────────┐    │
│  │   ASLTypingEngine           │    │  ← Debounce, auto-space, state
│  └────────────┬────────────────┘    │
│               │                      │
│  ┌────────────▼────────────────┐    │
│  │   get_suggestions()         │    │  ← Spell-check via pyenchant
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
         │
         │  JSON Response
         ▼
    Flutter Client
```

---

##  Getting Started

### Prerequisites

- Python 3.10+
- Docker (optional but recommended)

### Option 1: Run with Docker

```bash
# Clone the repository
git clone https://github.com/amr-ahmed-exe/ASL_API.git
cd ASL_API

# Build and run
docker build -t asl-api .
docker run -p 8000:8000 asl-api
```

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/amr-ahmed-exe/ASL_API.git
cd ASL_API

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

---

##  API Reference

### WebSocket Endpoint

**`WS /ws/predict`**

Establishes a real-time WebSocket connection for continuous ASL recognition.

#### Input — Per Frame (JSON)

```json
{
  "landmarks": [
    {"x": 0.52, "y": 0.73, "z": -0.04},
    ...
  ]
}
```

> 21 hand landmark points from MediaPipe Hands, one object per frame.

#### Control Commands (String)

| Command | Description |
|---|---|
| `"CLEAR"` | Resets the typing session |
| `"COMMIT:<word>"` | Commits a spell-check suggestion |

#### Output — Per Frame (JSON)

```json
{
  "status": "success",
  "raw_prediction": "H",
  "confidence": 0.91,
  "confirmed_letter": "H",
  "current_word": "HELLO",
  "final_word": "HELLO WORLD",
  "suggestions": ["HELLO", "HELP", "HELD", "HELM"]
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | `success` / `no_hand` / `invalid_landmarks_shape` |
| `raw_prediction` | string | Predicted letter this frame (A–Z) |
| `confidence` | float | Prediction confidence (0.0 – 1.0) |
| `confirmed_letter` | string / null | Letter confirmed after debounce |
| `current_word` | string | Word currently being signed |
| `final_word` | string | Full sentence so far |
| `suggestions` | array | Up to 4 spell-check suggestions |

---

##  Model Details

### Graph Convolutional Network (GCN)

The model treats the 21 MediaPipe hand landmarks as a **graph**, where nodes are joint positions and edges represent anatomical connections between fingers and the palm.

- **Input:** 21 × 3 (X, Y, Z) landmark coordinates → flattened to 63 floats
- **Architecture:** Multi-layer GCN with residual connections
- **Output:** Softmax over 26 classes (A–Z)
- **Checkpoint:** `sign_language_gcn_model.pth`

### Dynamic Letter Detection (J & Z)

Letters J and Z require motion, so they are handled separately using trajectory analysis:

- **J** → Pinky tip moves downward with a hook motion (from "I" base pose)
- **Z** → Index finger performs a zigzag with ≥2 direction changes along the X-axis

### Landmark Smoothing

Raw MediaPipe output contains frame-to-frame jitter. The `LandmarkFilter` class applies **Exponential Moving Average (EMA)** smoothing with `α = 0.4` to stabilize predictions.

### Typing Engine Parameters

| Parameter | Value | Description |
|---|---|---|
| `required_consecutive` | 5 frames | Frames needed to confirm a letter |
| `min_confidence` | 0.35 | Minimum confidence threshold |
| `repeat_delay` | 20 frames | Frames before same letter repeats |
| `no_hand_timeout` | 35 frames | Frames before auto-space is inserted |
| `history_size` | 7 frames | Rolling average window size |

---

##  Testing & Performance

| Test Type | Result |
|---|---|
| Sign Recognition Accuracy | **84.2%** (target: >80%) |
| Concurrent WebSocket Connections (load test) | **500** |
| Error Rate under 100 connections | **0%** |
| Test Cases (IEEE 829) | **30+** (unit, integration, WS, contract, E2E) |

---

##  Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI |
| **WebSocket Server** | Uvicorn + websockets |
| **ML Framework** | PyTorch (CPU & CUDA) |
| **Hand Landmarks** | MediaPipe (client-side) |
| **Spell Check** | pyenchant (en_US) |
| **Containerization** | Docker |
| **Mobile Client** | Flutter + Dart |
| **Cloud Deployment** | Azure VMSS |
| **CI/CD** | GitHub Actions |

---

##  Dependencies

```txt
fastapi==0.115.0
uvicorn==0.30.6
torch==2.2.2+cpu
numpy==1.26.4
pyenchant==3.2.2
python-multipart==0.0.9
websockets>=11.0.3
```

---

##  Team

| Name | Role |
|---|---|
| **Amr Ahmed El-Mokadam** | Team Lead · QA Manager · Backend Engineer |

---

##  License

Copyright © 2026 **Amr Ahmed**. All Rights Reserved.

---

<div align="center">

Made with ❤️ as a graduation project · Suez Canal University 2026

If you find this useful, please consider giving it a star!

</div>
