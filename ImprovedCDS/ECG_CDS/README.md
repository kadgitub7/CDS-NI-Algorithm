# CDS-ECG Edge Device — Arrhythmia Screening System

## Overview

This folder contains a complete pipeline for deploying the CDS-OVR (Class-Directed Splitting, One-vs-Rest) arrhythmia classifier as a real-time edge device. The system uses a cheap single-lead ECG sensor connected via Bluetooth to a phone/laptop running the classification algorithm.

**Classification task:** Normal sinus rhythm vs Abnormal (Atrial Fibrillation + Other rhythms)

**Algorithm:** CDS-OVR — a lightweight statistical classifier using supervised binning, Fisher-weighted feature scoring, and tree-structured subpopulation splitting. No neural networks, no GPU required.

---

## System Architecture

```
ECG Electrodes  -->  AD8232 Module  -->  ESP32-S3  ~~BLE~~>  Phone/Laptop
(3-lead pads)       (amplify+filter)    (ADC+stream)        (classify)
     ~$3                 ~$12               ~$10
                                                     Total: ~$22-25
```

**Data flow:**
1. Patient wears 3 adhesive ECG electrode pads (RA, LA, RL)
2. AD8232 ECG front-end IC amplifies and filters the analog signal
3. ESP32-S3 digitizes at 300Hz (12-bit ADC) and streams via BLE 5.0
4. Phone receives BLE packets, buffers a 30-second recording
5. 188 features are extracted (R-peak detection, HRV, morphology, frequency, entropy)
6. CDS-OVR classifier loads pre-trained model JSON and predicts Normal vs Abnormal
7. Result displayed with confidence scores

---

## File Inventory

### Core Algorithm (Training & Evaluation)
| File | Description |
|------|-------------|
| `cds_ovr_ecg.py` | CDS-OVR algorithm adapted for PhysioNet 2017 ECG data. 188 features, binary classification. Contains training, prediction, and evaluation code. |
| `physionet2017_feature_extraction_188.py` | Extracts 188 features from raw single-lead ECG waveforms (300Hz). 8 feature groups: AF evidence, RR statistics, Poincare, KDE, morphological, frequency, entropy, HR features. |
| `sweep_runner.py` | Parameter sweep runner v1 (strategies A-F). Tests different split features, preprocessing, and thresholds across 5 random seeds. |
| `sweep_v2.py` | Parameter sweep runner v2 (strategies G-L). Added rank-transform preprocessing, RATIO_EPS tuning, triple splits. |

### Edge Deployment Pipeline
| File | Description |
|------|-------------|
| `esp32_ad8232_ble.ino` | ESP32 Arduino firmware. Reads AD8232 analog output at 300Hz, streams 20-sample BLE packets. Includes leads-off detection. |
| `ble_ecg_receiver.py` | Python BLE client using `bleak`. Connects to ESP32, receives ECG stream, buffers recording. Includes `SimulatedReceiver` for hardware-free testing. |
| `model_export.py` | Trains CDS-OVR on the full PhysioNet 2017 dataset and exports all model artifacts (bin edges, class probabilities, tree structure, thresholds) to a JSON file. |
| `edge_classifier.py` | Inference-only engine. Loads exported model JSON, extracts 188 features from raw ECG, runs CDS-OVR prediction. No training code. |
| `run_device.py` | Main entry point. Connects to sensor (or simulator), records ECG, classifies, displays result. |
| `cds_model.json` | Exported trained model (0.12 MB). Contains everything needed for inference. |
| `requirements.txt` | Python dependencies. |

---

## Performance Results

### Algorithm Accuracy (PhysioNet 2017, 8,482 recordings)

Best configuration (Strategy J): dual split (AFEvidence + CVrr), rank-transform preprocessing.

| Metric | Value |
|--------|-------|
| Accuracy | 77.3% (mean across 5 seeds) |
| Best single seed | 78.7% |
| Specificity (Normal correct) | 80.7% |
| Sensitivity (Abnormal correct) | 72.1% |
| F1 (Abnormal class) | 71.6% |

### Complete Sweep Results (12 strategies, 5 seeds each, 90/10 split)

| Strategy | Mean Acc | Best Acc | Spec | Sens | F1 | Key Config |
|----------|----------|----------|------|------|----|------------|
| J (v2 winner) | 77.3% | 78.7% | 80.7% | 72.1% | 71.6% | Dual split + rank + RATIO_EPS=0.05 |
| D (v1 winner) | 77.6% | 78.7% | 83.2% | 69.0% | 70.9% | Dual split + z-score |
| A | 76.9% | 79.3% | 84.7% | 65.2% | 69.2% | AFEvidence + z-score |
| G | 76.9% | 78.3% | 77.8% | 75.7% | 72.3% | Dual split + rank |
| B | 76.8% | 79.4% | 88.8% | 58.7% | 66.7% | IrrEvidence + clip + 40 FPC |
| L | 76.6% | 77.7% | 74.3% | 80.1% | 73.2% | Triple + rank + tuned |
| F | 76.1% | 76.6% | 94.5% | 48.2% | 61.5% | Poincare + strict |
| C | 75.9% | 76.1% | 86.4% | 59.8% | 66.3% | CVrr + fine bins |
| H | 75.9% | 76.9% | 77.7% | 73.1% | 70.6% | Triple split + rank |
| E | 75.3% | 76.3% | 72.2% | 79.9% | 72.0% | MaxInfo 50 FPC |
| I | 75.1% | 76.0% | 71.7% | 80.3% | 71.9% | Winsorized + low threshold |
| K | 74.5% | 75.4% | 65.9% | 87.6% | 73.2% | Asymmetric against_scale=0.4 |

### Comparison vs Published Benchmarks

**PhysioNet 2017 Challenge (75 teams, hidden test set):**
- 4 teams tied for 1st: F1 = 0.83 (deep learning / large ensembles)
- Top 11 within F1 = 0.81-0.83
- Our CDS: F1 = 0.72 (gap of ~11 points)

**Context:**
- Challenge winners used deep CNNs and ensemble methods requiring GPUs
- CDS is a pure statistical algorithm — no matrix multiplications, no gradients, no GPU
- The original paper's claimed 95.4% on UCI used data leakage (global binning included test users in training). Honest LOOCV: 75.88%. Our PhysioNet result of 77% is consistent with what CDS truly achieves.
- CDS strengths: interpretable, lightweight (~0.12 MB model), runs on any device with Python

### End-to-End Pipeline Test Results

| Test | Input | Result | Classification Time |
|------|-------|--------|-------------------|
| Simulated Normal ECG | 30s synthetic sinus rhythm | Normal (ratio=2.0 vs threshold 1.8) | ~11s |
| Simulated AFib ECG | 30s synthetic irregular RR | Abnormal (ratio=12.9 vs threshold 1.8) | ~4s |
| Real PhysioNet A00001.mat | 30s real Normal recording | Normal (ratio=0.2 vs threshold 3.0) | ~6s |

---

## Hardware Setup Guide

### Shopping List

| Component | Approx Cost | Where to Buy |
|-----------|-------------|--------------|
| AD8232 ECG Sensor Module | $10-15 | Amazon, SparkFun (SEN-12650), AliExpress |
| ESP32-S3-DevKitC-1 board | $8-12 | Amazon, Mouser, AliExpress |
| Disposable ECG electrode pads (3-lead, bag of 30+) | $5-10 | Amazon ("3M Red Dot" or generic) |
| Jumper wires (female-to-female, 5 needed) | $3 | Amazon (usually comes in packs) |
| Micro-USB cable (for ESP32 power/programming) | $3 | Already have one likely |
| **Total** | **~$22-35** | |

### Wiring Diagram

```
AD8232 Module          ESP32-S3 Board
=============          ==============
OUTPUT  ------------>  GPIO 4  (analog input)
LO+     ------------>  GPIO 5  (leads-off detect +)
LO-     ------------>  GPIO 6  (leads-off detect -)
3.3V    ------------>  3V3
GND     ------------>  GND
SDN     (leave floating / not connected = active)
```

### ECG Electrode Placement (3-electrode, Lead I configuration)

```
     RA (white)              LA (black)
         O                       O
     right collarbone        left collarbone


                 RL (green/red)
                      O
              lower right abdomen
              (reference electrode)
```

- Clean skin with alcohol wipe before applying
- Ensure firm adhesion — loose pads cause noise
- Patient should sit still during recording
- Avoid placing over bone — find fleshy areas

### Flashing the ESP32

1. Install Arduino IDE (https://www.arduino.cc/en/software)
2. Add ESP32 board support:
   - File > Preferences > Additional Board Manager URLs:
   - Add: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
3. Tools > Board > Board Manager > Search "esp32" > Install "esp32 by Espressif"
4. Select board: Tools > Board > ESP32S3 Dev Module
5. Open `esp32_ad8232_ble.ino`
6. Connect ESP32 via USB, select correct COM port
7. Click Upload
8. Open Serial Monitor (115200 baud) to verify "BLE advertising started"

---

## Software Setup & Usage

### Installation

```bash
cd ImprovedCDS/ECG_CDS
pip install -r requirements.txt
```

### Step 1: Export the trained model (one-time)

This trains CDS-OVR on the full PhysioNet 2017 dataset and saves all model artifacts to JSON. Takes ~5 minutes.

```bash
python model_export.py
```

Output: `cds_model.json` (0.12 MB)

### Step 2: Test without hardware (simulated ECG)

```bash
# Simulate normal sinus rhythm
python run_device.py --sim normal

# Simulate atrial fibrillation
python run_device.py --sim afib

# Simulate other arrhythmia
python run_device.py --sim other
```

### Step 3: Test with saved PhysioNet recordings

```bash
# Classify a .mat file from the PhysioNet dataset
python run_device.py --file ../data/physioNetData2017/A00001.mat

# Classify a previously saved .npy recording
python run_device.py --file recordings/ecg_20260717_143000.npy
```

### Step 4: Run with real hardware

```bash
# Auto-scan for CDS-ECG-Sensor and start recording loop
python run_device.py

# Custom recording duration (default 30s)
python run_device.py --duration 45
```

The device will:
1. Scan Bluetooth for "CDS-ECG-Sensor"
2. Connect and wait for you to press Enter
3. Record for 30 seconds
4. Save the recording to `recordings/` folder
5. Extract 188 features and classify
6. Display Normal/Abnormal result with confidence scores
7. Loop back to step 2 (press Ctrl+C to quit)

---

## Testing Plan

### Phase 1: Software Validation (No hardware needed)

- [x] Model exports successfully from full dataset (8,482 records)
- [x] Simulated normal ECG classified as Normal
- [x] Simulated AFib ECG classified as Abnormal
- [x] Real PhysioNet .mat file classified correctly
- [ ] Run all PhysioNet test recordings through edge_classifier and compare accuracy to cds_ovr_ecg.py training results (should match since same model)
- [ ] Verify classification consistency: run same recording 3x, confirm identical results
- [ ] Test edge cases: very short recordings (<10s), noisy recordings, flat-line signals
- [ ] Measure classification latency on target device (phone/laptop)
- [ ] Verify model JSON loads correctly on different Python versions (3.9, 3.10, 3.11, 3.12, 3.13)

### Phase 2: Hardware Bring-Up (ESP32 + AD8232)

- [ ] Order hardware components (AD8232, ESP32-S3, electrode pads, jumper wires)
- [ ] Flash ESP32 with `esp32_ad8232_ble.ino`
- [ ] Verify serial output shows "BLE advertising started"
- [ ] Connect to ESP32 from phone/laptop using a BLE scanner app (e.g., nRF Connect) to verify the device is discoverable
- [ ] Wire AD8232 to ESP32 per wiring diagram
- [ ] Test leads-off detection: Serial Monitor should show "leads OFF" when electrodes not attached
- [ ] Attach electrodes to yourself, verify "leads ON" and serial output shows reasonable ADC values (not stuck at 0 or 4095)
- [ ] Use nRF Connect or similar to subscribe to ECG characteristic and verify data packets are arriving at ~300Hz

### Phase 3: End-to-End Integration

- [ ] Run `python ble_ecg_receiver.py` standalone — verify it finds and connects to the ESP32
- [ ] Record 30s of your own ECG using `python run_device.py`
- [ ] Verify saved `.npy` file contains reasonable data (plot it)
- [ ] Verify classification produces a result (Normal expected for healthy person)
- [ ] Record multiple sessions back-to-back — check for BLE connection stability
- [ ] Test at different distances (1m, 3m, 5m) — BLE range check
- [ ] Test with electrode removal mid-recording — verify leads-off warning appears

### Phase 4: Clinical Validation (Requires Medical Oversight)

- [ ] Collect ECG from 10+ healthy volunteers — all should classify as Normal
- [ ] If access to known AFib patients available: collect and verify Abnormal classification
- [ ] Compare classifications against physician interpretation of the same recordings
- [ ] Calculate real-world sensitivity/specificity on collected data
- [ ] Document false positive and false negative cases for analysis
- [ ] Review misclassified recordings to understand failure modes

### Phase 5: Deployment Polish

- [ ] Package as a standalone app (options below)
- [ ] Add real-time ECG waveform display during recording
- [ ] Add recording history and export to CSV/PDF
- [ ] Add patient ID / session metadata
- [ ] Battery life testing on ESP32 (estimate: 8-12 hours on 500mAh LiPo)
- [ ] Enclosure design for ESP32+AD8232 combo (3D print or off-the-shelf project box)

---

## Remaining Work & Next Steps

### Immediate (Before ordering hardware)

1. **Batch validation:** Run all 8,482 PhysioNet recordings through edge_classifier.py to confirm exported model matches training accuracy
2. **Optimize classification speed:** Current ~5-11 seconds per classification is acceptable but could be faster. Profile feature extraction to find bottlenecks (sample entropy and DFA are likely the slowest)
3. **Update cds_ovr_ecg.py with Strategy J parameters:** The winning sweep parameters have not been written back to the main algorithm file as defaults

### Short-Term (After hardware arrives)

4. **Hardware integration testing:** Follow Phase 2 and 3 of the testing plan above
5. **Signal quality assessment:** Compare AD8232 signal quality to PhysioNet recordings — the AD8232 is single-lead consumer-grade, PhysioNet used AliveCor KardiaMobile (also single-lead consumer-grade, same idea)
6. **BLE reliability:** Test connection stability over extended recording sessions

### Medium-Term (Product polish)

7. **Phone app options:**
   - **Kivy (Python -> Android APK):** Keeps everything in Python. Easiest path. Build with `buildozer`.
   - **Flutter + Python backend:** Better UI, but requires a local Python server on the phone (via Termux or Chaquopy).
   - **React Native + ONNX:** Convert model to ONNX for native inference. Requires rewriting feature extraction in JS/C++.
   - **Recommendation:** Start with Kivy for proof-of-concept, then evaluate Flutter if a polished UI is needed.

8. **Real-time ECG display:** Add a live waveform plot during recording using matplotlib (desktop) or Kivy Garden Graph (mobile)

9. **Recording management:** Save recordings with timestamps, patient IDs, and classification results in a local SQLite database

### Long-Term (If pursuing clinical use)

10. **Dataset expansion:** Train on PTB-XL (21,837 12-lead recordings, 71 diagnostic classes) for broader coverage. The CDS algorithm structure stays the same — only the feature extraction and number of classes change.

11. **Hybrid approach for better accuracy:** Use CDS for feature selection/ranking, then feed the top features into a lightweight random forest or logistic regression. Could push F1 from 0.72 to 0.80+ while keeping the model phone-friendly.

12. **Regulatory considerations:**
    - This is a screening tool, NOT a diagnostic device
    - For personal/research use: no regulatory approval needed
    - For clinical deployment: would need FDA 510(k) clearance (Class II medical device) or equivalent in other jurisdictions
    - De novo classification path may be available for AI-based screening tools
    - Labeling must clearly state "screening only, not for diagnosis"

13. **Multi-lead expansion:** The AD8232 is single-lead only. For 12-lead ECG:
    - ADS1298 (8-channel, ~$25) + ESP32 could do full 12-lead
    - Would enable PTB-XL compatibility and more diagnostic classes
    - Significantly more complex wiring and electrode placement

---

## Known Limitations

1. **Accuracy ceiling:** CDS-OVR achieves ~77% accuracy / 72% F1 on PhysioNet 2017. This is below challenge winners (F1=0.83) who used deep learning. CDS is best used as a screening filter, not a standalone diagnostic.

2. **Binary classification only:** Current model only distinguishes Normal vs Abnormal. It does not differentiate between AF, premature beats, flutter, or other specific arrhythmias.

3. **Single-lead constraint:** AD8232 provides Lead I only. Some arrhythmias are more visible in other leads (e.g., inferior MI in leads II/III/aVF).

4. **Feature extraction speed:** The 188-feature extraction involves sample entropy, DFA, EMD, and other computationally expensive algorithms. Classification takes 4-11 seconds on a modern laptop. On a phone, expect 10-30 seconds.

5. **Data leakage note:** The original CDS paper claimed 95.4% accuracy on UCI Arrhythmia dataset, but this used global binning where test users' features were included in training healthy ranges. Honest LOOCV yields 75.88%. Our results are consistent with the algorithm's true capability.

6. **AD8232 signal quality:** Consumer-grade ECG may have more noise than clinical equipment. The PhysioNet 2017 dataset was recorded with AliveCor KardiaMobile (also consumer-grade single-lead), so the training data is a reasonable match for our sensor.
