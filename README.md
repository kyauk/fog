# FoGStop: Real-Time Sensor Fusion for Gait Analysis

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-yellow.svg)]()

> **Optimizing a wearable inference engine from 73% to 90% sensitivity using multi-axis signal processing and data-driven debugging.**

## 🤖 Project Overview

This project implements a real-time **Freezing of Gait (FoG)** detection system for Parkinson's Disease patients. It functions as a **wearable edge-computing node**, processing high-frequency accelerometer data to detect gait anomalies with <10ms latency.

**Key Engineering Achievement:**
Improved detection sensitivity from **73% (baseline)** to **89.5%** by identifying critical signal processing flaws and implementing a multi-axis sensor fusion architecture, all while maintaining strict computational constraints for wearable deployment.

👉 **[Read the Engineering Retrospective: Learnings & Insights](LEARNINGS.md)**

---

## 🛠 Technical Highlights

### 1. Sensor Fusion & Signal Processing
Instead of treating sensor data as generic time-series, I applied physics-based feature extraction to capture the kinematics of freezing episodes.
- **Multi-Axis Fusion:** Fused X (Forward), Y (Vertical), and Z (Lateral) accelerometer streams to isolate the specific "lateral trembling" signature of FoG, which was previously lost in magnitude-only analysis.
- **Spectral Analysis:** Implemented FFT and Wavelet decomposition to analyze frequency shifts (3-8Hz freeze band) in real-time.
- **Digital Filtering:** Designed 4th-order Butterworth low-pass filters to remove sensor noise while preserving motion dynamics.

### 2. The "Normalization Bug": A Case Study in Data Physics
During development, I uncovered a critical bug where standard ML preprocessing (Z-score normalization) was applied *before* feature extraction.
- **The Problem:** Normalizing raw windows forced `mean=0` and `std=1`, mathematically destroying amplitude-based features like Energy and RMS.
- **The Fix:** Refactored the pipeline to extract features from raw filtered signals, preserving the physical magnitude of the motion.
- **Impact:** Recovered 4 critical features and improved sensitivity by **4.3%** instantly.
- **Lesson:** In physical systems (robotics/wearables), absolute signal magnitude carries information. Blindly applying ML preprocessing can erase physical reality.

### 3. Real-Time Inference Constraints
Designed for deployment on resource-constrained edge devices (e.g., Apple Watch).
- **Model Selection:** Chose **Random Forest** over Deep Learning (CNN/LSTM) to minimize inference latency and battery consumption.
- **Performance:** Achieved **89.5% Sensitivity** with **<10ms inference time** per window, proving that well-engineered features often outperform complex models in constrained environments.

---

## 📊 Performance Metrics

Evaluated using **Leave-One-Subject-Out (LOSO)** cross-validation on 17 patients to ensure generalization.

| Metric | FoGStop (Optimized) | IEEE Paper 1 (2024) | IEEE Paper 2 (2010) |
|--------|---------------------|---------------------|---------------------|
| **Sensitivity** | **89.5%** 🚀 | ~99% | 73.1% |
| **Specificity** | **78.5%** | ~99% | 81.6% |
| **Sensor Setup** | **Single Thigh (Wrist Proxy)** | Ankle (Optimal) | Waist/Thigh |
| **Features** | **148** (Multi-Axis) | 72 | 1 |

> **Note:** Achieving ~90% sensitivity with a single suboptimal sensor placement (thigh/wrist proxy) demonstrates the robustness of the feature engineering pipeline.

---

## 🧠 System Architecture

### Pipeline
1.  **Input:** 64Hz 3-axis Accelerometer Data
2.  **Preprocessing:** 
    - 4th-order Butterworth Filter (20Hz cutoff)
    - Sliding Window (4s window, 10% overlap)
3.  **Feature Engineering (148 Features):**
    - **Time-Domain:** RMS, Jerk, Crest Factor, Zero-Crossing Rate
    - **Frequency-Domain:** Spectral Entropy, Dominant Frequency, Power Band Ratios
    - **Wavelet:** Discrete Wavelet Transform (DWT) Energy
4.  **Inference:** Random Forest Classifier (200 trees)
5.  **Output:** Binary Classification (Freeze / No Freeze)

### Code Structure
```bash
├── src/
│   ├── feature_extractor.py  # Physics-based feature engineering
│   ├── preprocess.py         # Digital signal processing pipeline
│   ├── MLModel.py            # Random Forest implementation & LOSO validation
│   └── visualizer.py         # Signal visualization tools
├── main.py                   # End-to-end training & validation pipeline
└── scripts/
    └── diagnose.py           # System diagnostic tools
```

---

## 🚀 Future Roadmap

- **CoreML Integration:** Porting the trained Random Forest to CoreML for on-device inference on Apple Watch.
- **Quantization:** Reducing model precision to FP16/INT8 to further optimize for embedded microcontrollers.
- **Kalman Filtering:** Implementing state estimation to smooth trajectory tracking and reduce false positives.

---

## 👨‍💻 Author
**Jason Kyauk**
