# 🧠 Engineering Learnings & Insights

> **"The difference between 73% and 90% sensitivity wasn't a better model—it was better physics."**

This document details the key engineering insights gained during the optimization of the FoGStop system. It serves as a technical retrospective on signal processing, debugging, and real-time system design.

---

## 1. The "Normalization Bug": A Lesson in Data Physics

**The Incident:**
Early in the project, the model was stuck at ~82% sensitivity. Despite adding advanced features (spectral entropy, wavelets), performance wouldn't budge.

**The Debugging Process:**
I wrote a custom diagnostic script (`scripts/diagnose.py`) to inspect the feature distribution. I discovered that 4 critical features (`mean`, `variance`, `energy`, `rms`) had **zero variance** across the entire dataset.

**The Root Cause:**
The preprocessing pipeline was applying Z-score normalization (`(x - μ) / σ`) to the raw signal windows *before* feature extraction.
- **Mathematical Consequence:** Every window was forced to have `mean=0` and `std=1`.
- **Physical Consequence:** The absolute magnitude of the acceleration (which indicates movement intensity) was mathematically erased. A violent tremor and a gentle sway looked identical to the feature extractor.

**The Fix:**
I refactored the pipeline to extract features from the **raw filtered signal**, preserving the physical units (g-force). Normalization was moved to *after* feature extraction (and ultimately removed as Random Forest is scale-invariant).

**Takeaway:**
In robotics and physical systems, **absolute magnitude matters**. Blindly applying standard ML preprocessing (like normalization) without understanding the physical sensor data can destroy critical information.

---

## 2. Sensor Fusion: Capturing Lateral Dynamics

**The Hypothesis:**
Freezing of Gait isn't just "stopping"—it's a high-frequency trembling. But trembling in which direction?

**The Analysis:**
The initial model used only the **Signal Magnitude Vector (SMV)** (`sqrt(x^2 + y^2 + z^2)`). While efficient, SMV destroys directional information. A lateral (side-to-side) tremor looks the same as a vertical step impact in the magnitude domain.

**The Solution:**
I expanded the feature extraction engine to process **X (Forward), Y (Vertical), and Z (Lateral)** axes independently, plus the Magnitude.
- **Result:** Feature count increased from 37 to 148.
- **Performance:** Sensitivity jumped from **86.6% to 89.5%**.
- **Why:** The model learned to identify "High Lateral Variance + Low Vertical Variance" as a specific signature of Freezing.

**Takeaway:**
**Sensor Fusion > Model Complexity.** Instead of throwing a Deep Learning model at the magnitude data, simply exposing the raw axis data allowed a simpler model (Random Forest) to learn the correct physical dynamics.

---

## 3. Real-Time Constraints vs. Model Complexity

**The Constraint:**
The system targets deployment on an Apple Watch (Series 6+), with a strict latency budget of <50ms per window to allow for real-time cueing.

**The Trade-off:**
- **Deep Learning (CNN/LSTM):** High accuracy, but high latency (inference + data loading) and high battery drain.
- **Random Forest:** Lower theoretical capacity, but **microsecond inference** and interpretable decision paths.

**The Decision:**
I stuck with Random Forest and focused on **Feature Engineering**. By crafting 148 physics-based features (FFT, Wavelets), I achieved **89.5% sensitivity**—comparable to Deep Learning benchmarks—with a fraction of the compute cost.

**Takeaway:**
In embedded systems (robotics/wearables), **feature engineering is the most efficient form of compute**. Pre-calculating smart features is often cheaper and more robust than running a massive neural network.

---

## 4. Summary of Skills Applied

- **Signal Processing:** FFT, Butterworth Filtering, Wavelet Transform.
- **Sensor Fusion:** Multi-axis accelerometer integration.
- **Data-Driven Debugging:** Custom diagnostic tooling to validate data integrity.
- **Embedded ML Mindset:** Prioritizing inference latency and compute efficiency.
