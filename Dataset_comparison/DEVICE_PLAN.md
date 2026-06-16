# Wearable CDS-ML Arrhythmia Detection Device — Build Plan

## Overview

A wearable, real-time arrhythmia detection system that implements the hybrid
CDS + Neural Network algorithm on hardware. Two independent tracks are
presented — choose one based on your skills and goals, or build both and
compare.

```
                    ┌──────────────────────────────┐
                    │   SHARED: Sensors + Algorithm │
                    │   MAX86150 ECG/PPG            │
                    │   Feature extraction          │
                    │   CDS Alg 4 + ANN/ESN logic   │
                    └──────────┬───────────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
     ┌────────▼─────────┐            ┌──────────▼────────┐
     │  TRACK A: EMBEDDED│            │  TRACK B: FPGA    │
     │  Arduino / ARM    │            │  Verilog / HW     │
     │  C/C++ software   │            │  Parallel logic   │
     │  Faster prototype │            │  Lower power      │
     │  ~$60 BOM         │            │  ~$90 BOM         │
     └───────────────────┘            └───────────────────┘
```

Inspired by:
- Tele-Health COVID-19 monitoring system (Alam et al., 2021)
- PPG-based BP estimation meta-analysis (IEEE, 2024)
- FPGA-based ECG arrhythmia classifiers (ETASR 2025, MDPI 2025)

---

## SHARED: SENSORS (both tracks use the same sensors)

### Core Sensors

| Sensor | Part | Measures | Interface | Cost |
|--------|------|----------|-----------|------|
| ECG + PPG | **MAX86150** | Single-lead ECG + red/IR PPG | I2C/SPI | $8 |
| Temperature | **MAX30205** | Skin temp ±0.1°C | I2C | $3 |
| IMU | **BMI160** | 3-axis accel + gyro | SPI/I2C | $4 |

### Shared Signal Processing Pipeline

Both tracks implement the same signal chain — only the language differs
(C for Track A, Verilog for Track B).

```
Raw ECG (200 Hz) ──► BPF 0.5-40 Hz ──► Notch 50/60 Hz ──► R-peak detect
                                                                │
Raw PPG (100 Hz) ──► Detrend ──► BPF 0.5-5 Hz ──► Peak detect  │
                                                       │        │
                                              SpO2 calc │   PTT calc
                                                       ▼        ▼
                                              ┌──────────────────┐
                                              │ Feature Vector   │
                                              │ (~45 features)   │
                                              └────────┬─────────┘
                                                       │
                                              ┌────────▼─────────┐
                                              │ CDS Alg 4        │
                                              │ + ML Classifier   │
                                              │ (Cascade hybrid)  │
                                              └────────┬─────────┘
                                                       │
                                                   Decision
                                            (Healthy/Screening/Alarm)
```

### Feature Extraction (maps to UCI arrhythmia features)

| Feature Group | Count | Source |
|--------------|-------|--------|
| Demographics | 4 | User profile (age, sex, height, weight) |
| Intervals | 5 | QRS duration, PR, QT, T, P intervals from ECG |
| Angles & rate | 6 | QRS/T/P angles, heart rate |
| Waveform morphology | ~22 | Q/R/S widths, amplitudes from single-lead |
| PPG-derived | 8 | SpO2, PTT, pulse rate, BP est, resp rate, HRV |
| **Total** | **~45** | |

CDS Algorithm 3 prunes ~90% — the device only needs ~20-30 features.

### Algorithm on Device (both tracks)

**Offline (PC, done once):** Train Algorithms 1-3, export:
- Tree structure (~500 bytes)
- Healthy ranges per (node, feature): 720 bytes
- Action weights: 1,440 bytes
- **Total model size: ~2.7 KB**

**On device (real-time):** Algorithm 4 lookup + ML inference.

---

# ========================================================================
# TRACK A: EMBEDDED (Arduino / ARM / C++)
# ========================================================================

## A1. Platform Selection

| Option | Board | CPU | RAM | Flash | BLE | Cost | Best for |
|--------|-------|-----|-----|-------|-----|------|----------|
| **A1a: Arduino** | Arduino Nano 33 BLE Sense | nRF52840 (64 MHz Cortex-M4F) | 256 KB | 1 MB | BLE 5.0 built-in | $33 | Fastest start, huge community |
| **A1b: ESP32** | ESP32-S3 DevKit | Xtensa LX7 (240 MHz, dual-core) | 512 KB | 8 MB | BLE 5.0 + WiFi | $10 | Cheapest, WiFi option, more RAM |
| **A1c: Nordic** | nRF52840 DK | Cortex-M4F (64 MHz) | 256 KB | 1 MB | BLE 5.0 | $40 | Best BLE SDK, lowest power |
| **A1d: STM32** | Nucleo-L476RG | Cortex-M4F (80 MHz) | 128 KB | 1 MB | External module | $15 | Best peripherals, industrial |

**Recommendation**: **Arduino Nano 33 BLE Sense** for prototyping (nRF52840
with BLE built-in, Arduino IDE support, I2C/SPI ready). Move to bare nRF52840
or ESP32-S3 for the final PCB.

## A2. System Architecture

```
┌────────────────────────────────────────────────────────────┐
│  TRACK A: EMBEDDED WEARABLE                                │
│                                                             │
│  ┌──────────┐   I2C    ┌───────────────────────────────┐   │
│  │ MAX86150 │─────────▶│                               │   │
│  │ ECG+PPG  │          │   Arduino Nano 33 BLE Sense   │   │
│  └──────────┘          │   (nRF52840, 64 MHz, 256KB)   │   │
│  ┌──────────┐   I2C    │                               │   │
│  │ MAX30205 │─────────▶│   ┌───────────────────────┐   │   │
│  │ Temp     │          │   │ Software layers:      │   │   │
│  └──────────┘          │   │                       │   │   │
│  ┌──────────┐   SPI    │   │ 1. Sensor drivers     │   │   │
│  │ BMI160   │─────────▶│   │ 2. DSP filters (C)    │   │   │
│  │ IMU      │          │   │ 3. Feature extraction  │   │   │
│  └──────────┘          │   │ 4. CDS Alg4 lookup    │   │   │
│                         │   │ 5. ANN inference      │   │   │
│  ┌──────────┐          │   │ 6. Cascade decision    │   │   │
│  │ LiPo     │ 3.7V     │   │ 7. BLE transmit       │   │   │
│  │ 500 mAh  │─────────▶│   └───────────────────────┘   │   │
│  └──────────┘          │                               │   │
│  ┌──────────┐          │   BLE 5.0 ────────────────────┼──▶ Phone
│  │ Vibration│◀─────────│                               │   │
│  │ Motor    │          └───────────────────────────────┘   │
│  └──────────┘                                              │
└────────────────────────────────────────────────────────────┘
```

## A3. Software Architecture (C/C++)

```
project/
├── src/
│   ├── main.cpp                 // Arduino setup() + loop()
│   ├── sensors/
│   │   ├── max86150_driver.h    // ECG + PPG register config, raw read
│   │   ├── max30205_driver.h    // Temperature I2C
│   │   └── bmi160_driver.h      // IMU SPI
│   ├── dsp/
│   │   ├── butterworth.h        // IIR bandpass/notch filters
│   │   ├── rpeak_detect.h       // Pan-Tompkins R-peak detector
│   │   ├── ppg_process.h        // SpO2, systolic peak, PTT
│   │   └── feature_extract.h    // Compute 45-feature vector
│   ├── algorithm/
│   │   ├── cds_model.h          // CDS tree + ranges (const arrays)
│   │   ├── cds_alg4.h           // Algorithm 4 range check + AF
│   │   ├── ann_weights.h        // ANN weights (const arrays)
│   │   ├── ann_inference.h      // Forward pass: matmul + ReLU
│   │   └── cascade_decision.h   // CDS → ML routing logic
│   └── comms/
│       ├── ble_service.h        // BLE GATT service definition
│       └── ble_protocol.h       // Packet encoding
├── model_export/
│   ├── export_cds_model.py      // Python → C header converter
│   └── export_ann_weights.py    // Python → C header converter
└── platformio.ini               // Build config
```

## A4. Key Implementation Details

### CDS Algorithm 4 in C

```c
// cds_alg4.h — Algorithm 4 as a simple lookup
#include "cds_model.h"  // auto-generated from Python

typedef enum { HEALTHY, SCREENING, UNHEALTHY } CDS_Decision;

typedef struct {
    CDS_Decision decision;
    float        af;           // assurance factor
    int          alarm_feature; // -1 if no alarm
} CDS_Result;

CDS_Result cds_predict(const float features[], int node_idx) {
    CDS_Result result = { SCREENING, 0.0f, -1 };
    float af = 0.0f;

    for (int i = 0; i < N_RETAINED_FEATURES; i++) {
        int f = retained_features[i];
        float val = features[f];
        float bmin = healthy_ranges[node_idx][i][0];
        float bmax = healthy_ranges[node_idx][i][1];

        if (val < bmin || val > bmax) {
            // ALARM — feature outside healthy range
            result.decision = UNHEALTHY;
            result.alarm_feature = f;
            return result;
        }

        // Accumulate AF (Eq. 7)
        for (int h = 0; h < N_DISEASE_CLASSES; h++) {
            float p_h_f = class_probs[node_idx][h];
            float r_oh  = action_weights[node_idx][i][h];
            float p_d_f = disease_prob[node_idx];
            if (p_d_f > 0.0f) {
                af += p_h_f * r_oh / p_d_f;
            }
        }
    }

    result.af = af;
    float rw = 1.0f - af;
    if (rw <= DIAGNOSTIC_THRESHOLD) {
        result.decision = HEALTHY;
    }
    return result;
}
```

### ANN Forward Pass in C

```c
// ann_inference.h — tiny MLP forward pass
#include "ann_weights.h"  // auto-generated: W1[32][45], b1[32], W2[2][32], b2[2]

void ann_predict(const float input[45], float output[2]) {
    float hidden[32];

    // Layer 1: input(45) → hidden(32), ReLU
    for (int j = 0; j < 32; j++) {
        float sum = b1[j];
        for (int i = 0; i < 45; i++) {
            sum += W1[j][i] * input[i];   // 45 × 32 = 1,440 multiplications
        }
        hidden[j] = (sum > 0.0f) ? sum : 0.0f;  // ReLU
    }

    // Layer 2: hidden(32) → output(2), softmax
    float max_val = -1e9f;
    for (int j = 0; j < 2; j++) {
        float sum = b2[j];
        for (int i = 0; i < 32; i++) {
            sum += W2[j][i] * hidden[i];  // 32 × 2 = 64 multiplications
        }
        output[j] = sum;
        if (sum > max_val) max_val = sum;
    }

    // Softmax
    float exp_sum = 0.0f;
    for (int j = 0; j < 2; j++) {
        output[j] = expf(output[j] - max_val);
        exp_sum += output[j];
    }
    for (int j = 0; j < 2; j++) {
        output[j] /= exp_sum;
    }
}
// Total: 1,504 multiplications per inference
```

### Model Export Script (Python → C headers)

```python
# export_ann_weights.py — run once after training
# Reads trained sklearn MLP, writes C header with weight arrays

def export_to_c_header(clf, scaler, filepath="ann_weights.h"):
    W1 = clf.coefs_[0].T   # (32, 45)
    b1 = clf.intercepts_[0] # (32,)
    W2 = clf.coefs_[1].T   # (2, 32)
    b2 = clf.intercepts_[1] # (2,)

    with open(filepath, "w") as f:
        f.write("// Auto-generated ANN weights\n")
        f.write(f"#define ANN_INPUT_SIZE 45\n")
        f.write(f"#define ANN_HIDDEN_SIZE 32\n")
        f.write(f"#define ANN_OUTPUT_SIZE 2\n\n")
        _write_array_2d(f, "W1", W1)
        _write_array_1d(f, "b1", b1)
        _write_array_2d(f, "W2", W2)
        _write_array_1d(f, "b2", b2)
```

## A5. Power Budget (Track A)

| Component | Active | Duty cycle | Average |
|-----------|--------|------------|---------|
| MAX86150 ECG+PPG | 1.5 mA | 100% | 1.50 mA |
| nRF52840 processing | 5.0 mA | 30% | 1.50 mA |
| nRF52840 BLE TX | 8.0 mA | 5% | 0.40 mA |
| BMI160 IMU | 0.9 mA | 10% | 0.09 mA |
| MAX30205 temp | 0.04 mA | 1% | 0.00 mA |
| Vibration motor | 50 mA | 0.1% | 0.05 mA |
| **Total** | | | **3.54 mA** |

Battery life: 500 mAh / 3.54 mA = **141 hours (~6 days)**

## A6. Bill of Materials (Track A)

| Component | Part | Cost |
|-----------|------|------|
| Arduino Nano 33 BLE Sense | nRF52840 + 9-axis IMU | $33 |
| ECG+PPG sensor | MAX86150 breakout | $8 |
| Temperature | MAX30205 breakout | $3 |
| LiPo battery | 500 mAh 3.7V | $5 |
| Vibration motor | 3V coin type | $1 |
| PCB fabrication | JLCPCB 2-layer, 5 pcs | $10 |
| Enclosure + strap | 3D printed + elastic | $5 |
| Misc (connectors, caps, resistors) | | $5 |
| **Total** | | **~$70** |

## A7. Timeline (Track A)

```
Week  1-2:  Order parts. Set up Arduino IDE + libraries.
            Wire MAX86150 to Arduino on breadboard.
            Verify raw ECG + PPG readings on Serial Plotter.

Week  3-4:  Implement DSP filters (butterworth.h, rpeak_detect.h).
            Validate R-peak detection against PhysioNet MIT-BIH.
            Implement PPG processing (SpO2, PTT).

Week  5-6:  Implement feature extraction (45 features from ECG+PPG).
            Run export_cds_model.py and export_ann_weights.py.
            Implement CDS Alg4 in C. Validate vs Python output.

Week  7-8:  Implement ANN forward pass in C.
            Implement cascade decision logic.
            End-to-end test: raw signal → decision on Serial Monitor.

Week  9-10: Implement BLE service. Build basic phone app (Flutter/React Native).
            Stream vitals + decisions to phone.

Week 11-12: Design custom PCB (KiCad). Order from JLCPCB.
            3D print enclosure. Assemble prototype.

Week 13-14: Bench test with ECG simulator.
            Power measurement. Battery life test.

Week 15-16: Human pilot test (10 healthy volunteers).
            Iterate on comfort, signal quality, false alarm rate.
```

## A8. Prototyping Stages (Track A)

```
Stage 1: BREADBOARD                    Stage 2: PERFBOARD
┌─────────────────────┐               ┌─────────────────────┐
│ Arduino ──► Serial  │               │ Arduino + sensors   │
│ MAX86150 on jumpers │               │ soldered, battery   │
│ Laptop display      │               │ BLE → phone app     │
│ No battery          │               │ Velcro chest mount  │
└─────────────────────┘               └─────────────────────┘

Stage 3: CUSTOM PCB                    Stage 4: FINAL PRODUCT
┌─────────────────────┐               ┌─────────────────────┐
│ All-in-one PCB      │               │ Injection-molded    │
│ nRF52840 bare chip  │               │ enclosure           │
│ Compact form factor │               │ Medical-grade leads │
│ LiPo + charger IC   │               │ CE/FDA prep         │
└─────────────────────┘               └─────────────────────┘
```

---

# ========================================================================
# TRACK B: FPGA (Verilog / Hardware Description)
# ========================================================================

## B1. Platform Selection

| Option | Board | FPGA | LUTs | BRAM | Cost | Best for |
|--------|-------|------|------|------|------|----------|
| **B1a: iCE40** | Lattice iCEstick / UPduino | iCE40 UP5K | 5,280 | 120 Kbit | $15-25 | Ultra low power, open-source toolchain |
| **B1b: ECP5** | OrangeCrab / Colorlight | ECP5-25F | 24,000 | 1,008 Kbit | $30-50 | More room, DSP blocks, USB |
| **B1c: Xilinx** | PYNQ-Z2 / Cmod A7 | Artix-7 | 20,800+ | 1,800 Kbit | $70-120 | Most DSP slices, Vivado ecosystem |
| **B1d: Gowin** | Tang Nano 9K | GW1NR-9C | 8,640 | 468 Kbit | $15 | Cheap, good beginner FPGA |

**Recommendation**: **Tang Nano 9K** for prototyping (cheap, 8.6K LUTs, USB
programmer, open-source Gowin EDA). Move to **iCE40 UP5K** for the final
wearable (lowest power: ~75 μW/MHz).

For BLE: FPGA boards don't have BLE built-in. Add an **nRF52832 module** ($5)
connected via UART/SPI as a BLE radio.

## B2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  TRACK B: FPGA WEARABLE                                             │
│                                                                      │
│  ┌──────────┐  I2C  ┌──────────────────────────────────────────┐    │
│  │ MAX86150 │──────▶│  FPGA (Tang Nano 9K / iCE40 UP5K)       │    │
│  │ ECG+PPG  │       │                                          │    │
│  └──────────┘       │  ┌──────────┐  ┌──────────┐  ┌───────┐  │    │
│  ┌──────────┐  I2C  │  │ I2C/SPI  │  │  Signal  │  │Feature│  │    │
│  │ MAX30205 │──────▶│  │ Sensor   │─▶│  Process │─▶│Extract│  │    │
│  │ Temp     │       │  │ Interface│  │  (DSP)   │  │       │  │    │
│  └──────────┘       │  └──────────┘  └──────────┘  └───┬───┘  │    │
│  ┌──────────┐  SPI  │                                   │      │    │
│  │ BMI160   │──────▶│  ┌──────────┐  ┌──────────┐      │      │    │
│  │ IMU      │       │  │  CDS     │◀─┘           │      │      │    │
│  └──────────┘       │  │  Range   │   ┌──────────▼─┐   │      │    │
│                      │  │  Check   │   │  ANN / ESN  │   │      │    │
│  ┌──────────┐       │  │ (2 clk)  │   │  Forward    │   │      │    │
│  │ nRF52832 │◀─UART─│  └────┬─────┘   │  Pass       │   │      │    │
│  │ BLE      │       │       │          │ (870 clk)   │   │      │    │
│  │ Module   │──────▶│  ┌────▼──────────▼───────────┐ │   │      │    │
│  └──────────┘       │  │    CASCADE DECISION       │ │   │      │    │
│      │  BLE         │  │    (1 clk combinational)  │ │   │      │    │
│      ▼              │  └───────────┬───────────────┘ │   │      │    │
│   Phone App         │              │ UART TX          │   │      │    │
│                      └──────────────┴──────────────────┘   │      │    │
│  ┌──────────┐                                              │      │    │
│  │ LiPo     │ 3.7V, 250 mAh                               │      │    │
│  │ Battery  │ (~100+ hrs)                                   │      │    │
│  └──────────┘                                              │      │    │
└──────────────────────────────────────────────────────────────────────┘
```

## B3. Verilog Module Hierarchy

```
rtl/
├── top.v                        // Top-level: clocks, resets, I/O routing
├── sensors/
│   ├── i2c_master.v             // I2C controller for MAX86150 + MAX30205
│   ├── spi_master.v             // SPI controller for BMI160
│   ├── max86150_ctrl.v          // Register config, FIFO read, raw sample out
│   └── sensor_mux.v             // Round-robin sample collection
├── dsp/
│   ├── iir_biquad.v             // Single biquad section (parameterized)
│   ├── bandpass_filter.v        // Chain of biquads: 0.5-40 Hz ECG
│   ├── notch_filter.v           // 50/60 Hz notch
│   ├── rpeak_detector.v         // Adaptive threshold R-peak detect
│   ├── ppg_processor.v          // SpO2 ratio, systolic peak, PTT
│   └── feature_pipeline.v       // Extracts 45-feature vector over window
├── algorithm/
│   ├── cds_range_check.v        // Parallel range compare (all features, 1 clk)
│   ├── cds_af_accumulator.v     // Sequential AF summation (Eq. 7)
│   ├── ann_mac_unit.v           // Multiply-accumulate (shared or parallel)
│   ├── ann_layer.v              // One dense layer: MAC array + ReLU
│   ├── ann_forward.v            // 2-layer ANN: layer1 → layer2 → softmax
│   ├── esn_reservoir.v          // (alternative) ESN: PRNG weights + tanh LUT
│   ├── cascade_decision.v       // CDS result + ML result → final decision
│   └── model_rom.v              // Block RAM storing weights + ranges
├── comms/
│   ├── uart_tx.v                // UART to nRF52832 BLE module
│   └── packet_encoder.v         // Decision + vitals → byte stream
├── memory/
│   ├── weight_rom.v             // ANN weights in BRAM (auto-generated)
│   └── range_rom.v              // CDS healthy ranges in BRAM (auto-generated)
└── tb/
    ├── top_tb.v                 // Full system testbench
    ├── ecg_stimulus.v           // Replay PhysioNet data as test input
    └── python_checker.py        // Compare Verilog output vs Python golden ref
```

## B4. Key Verilog Modules

### CDS Range Check (parallel, 1 clock cycle)

```verilog
module cds_range_check #(
    parameter N_FEATURES = 30
)(
    input  wire                          clk,
    input  wire                          valid_in,
    input  wire signed [15:0]            features    [0:N_FEATURES-1],  // Q8.8
    input  wire signed [15:0]            b_min       [0:N_FEATURES-1],  // Q8.8
    input  wire signed [15:0]            b_max       [0:N_FEATURES-1],  // Q8.8
    output reg                           alarm,
    output reg  [7:0]                    alarm_feature_idx,
    output reg                           all_in_range,
    output reg                           valid_out
);
    integer i;
    reg alarm_found;
    reg [7:0] first_alarm;

    always @(posedge clk) begin
        if (valid_in) begin
            alarm_found = 1'b0;
            first_alarm = 8'd0;
            for (i = 0; i < N_FEATURES; i = i + 1) begin
                if (!alarm_found &&
                    (features[i] < b_min[i] || features[i] > b_max[i])) begin
                    alarm_found = 1'b1;
                    first_alarm = i[7:0];
                end
            end
            alarm             <= alarm_found;
            alarm_feature_idx <= first_alarm;
            all_in_range      <= ~alarm_found;
            valid_out         <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end
endmodule
```

### ANN MAC Unit (pipelined multiply-accumulate)

```verilog
module ann_mac_unit #(
    parameter DATA_WIDTH = 16  // Q8.8 fixed-point
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          start,
    input  wire signed [DATA_WIDTH-1:0]  weight,
    input  wire signed [DATA_WIDTH-1:0]  activation,
    input  wire signed [31:0]            acc_in,
    output reg  signed [31:0]            acc_out,
    output reg                           done
);
    always @(posedge clk) begin
        if (rst) begin
            acc_out <= 32'd0;
            done    <= 1'b0;
        end else if (start) begin
            // Q8.8 × Q8.8 = Q16.16, accumulate in Q16.16
            acc_out <= acc_in + (weight * activation);
            done    <= 1'b1;
        end else begin
            done <= 1'b0;
        end
    end
endmodule
```

### ANN Forward Pass (sequential, reuses MAC unit)

```verilog
module ann_forward #(
    parameter INPUT_SIZE  = 45,
    parameter HIDDEN_SIZE = 32,
    parameter OUTPUT_SIZE = 2
)(
    input  wire        clk, rst, start,
    input  wire [15:0] features [0:INPUT_SIZE-1],
    output reg  [15:0] output_probs [0:OUTPUT_SIZE-1],
    output reg         valid_out,
    output reg  [1:0]  predicted_class
);
    // Weight ROMs
    // W1: HIDDEN_SIZE × INPUT_SIZE  stored in BRAM
    // W2: OUTPUT_SIZE × HIDDEN_SIZE stored in BRAM

    // State machine: IDLE → LAYER1 → RELU → LAYER2 → SOFTMAX → DONE
    // Reuses a single MAC unit, iterating over weights
    // Total clocks: INPUT_SIZE × HIDDEN_SIZE + HIDDEN_SIZE × OUTPUT_SIZE
    //             = 45×32 + 32×2 = 1,504 clock cycles
    // At 12 MHz: 125 μs

    // (Full implementation omitted for brevity —
    //  FSM feeds weight+activation pairs into ann_mac_unit sequentially,
    //  stores hidden activations in register file, applies ReLU,
    //  then computes output layer the same way)
endmodule
```

## B5. Fixed-Point Arithmetic

All computations use **Q8.8** (16-bit signed fixed-point):

| Property | Value |
|----------|-------|
| Total bits | 16 (1 sign + 7 integer + 8 fraction) |
| Range | -128.000 to +127.996 |
| Resolution | 0.00390625 (1/256) |
| Multiplication result | Q16.16 (32-bit), truncate back to Q8.8 |

Sufficient for: ECG amplitudes (±30 mV), angles (-180 to +180), all CDS ranges.

## B6. FPGA Resource Budget

| Module | LUTs | BRAM bits | Clocks/decision |
|--------|------|-----------|-----------------|
| I2C master | 200 | 0 | - |
| SPI master | 150 | 0 | - |
| Sensor controller | 300 | 0 | - |
| IIR filters (×4 biquads) | 800 | 0 | streaming |
| R-peak detector | 300 | 2,048 | streaming |
| Feature pipeline | 500 | 4,096 | 200 Hz |
| CDS range check | 400 | 3,840 | 1 clk |
| CDS AF accumulator | 200 | 1,440 | 30 clks |
| ANN forward pass | 600 | 12,032 | 1,504 clks |
| Cascade decision | 50 | 0 | 1 clk |
| UART TX | 100 | 0 | - |
| Top-level glue | 200 | 0 | - |
| **Total** | **3,800** | **23,456** | **~1,536** |
| **Available (Tang Nano 9K)** | **8,640** | **468K** | |
| **Utilization** | **44%** | **5%** | |

At 12 MHz: 1,536 clocks = **128 μs per decision** (7,800 decisions/second).

## B7. Power Budget (Track B)

| Component | Active | Duty cycle | Average |
|-----------|--------|------------|---------|
| MAX86150 ECG+PPG | 1.5 mA | 100% | 1.50 mA |
| FPGA core | 1.0 mA | 20% | 0.20 mA |
| FPGA I/O | 0.5 mA | 20% | 0.10 mA |
| nRF52832 BLE module | 6.0 mA | 5% | 0.30 mA |
| BMI160 IMU | 0.9 mA | 10% | 0.09 mA |
| MAX30205 temp | 0.04 mA | 1% | 0.00 mA |
| **Total** | | | **2.19 mA** |

Battery life: 250 mAh / 2.19 mA = **114 hours (~4.7 days)**
             500 mAh / 2.19 mA = **228 hours (~9.5 days)**

## B8. Bill of Materials (Track B)

| Component | Part | Cost |
|-----------|------|------|
| FPGA board | Tang Nano 9K | $15 |
| BLE module | nRF52832 module (Raytac MDBT42Q) | $5 |
| ECG+PPG sensor | MAX86150 breakout | $8 |
| Temperature | MAX30205 breakout | $3 |
| IMU | BMI160 breakout | $4 |
| LiPo battery | 500 mAh 3.7V | $5 |
| LDO regulator | 3.3V, 300mA (MCP1700) | $1 |
| PCB fabrication | JLCPCB 2-layer, 5 pcs | $10 |
| Enclosure + strap | 3D printed + elastic | $5 |
| Misc (level shifters, caps, connectors) | | $10 |
| **Total** | | **~$66** |

## B9. Timeline (Track B)

```
Week  1-2:  Order Tang Nano 9K + sensors. Set up Gowin EDA / Yosys toolchain.
            Blink LED, verify UART TX works.

Week  3-4:  Implement I2C master. Read raw MAX86150 ECG+PPG samples.
            Display on PC via UART. Verify signal quality.

Week  5-6:  Implement IIR biquad filter in Verilog. Chain into bandpass.
            Implement R-peak detector. Validate vs PhysioNet golden data.

Week  7-8:  Implement feature extraction pipeline.
            Feed known UCI feature vectors → verify CDS range check.
            Implement CDS AF accumulator.

Week  9-10: Implement ANN forward pass (MAC unit + layer FSM).
            Export weights from Python to weight_rom.v.
            Implement cascade decision logic.

Week 11-12: Integrate nRF52832 BLE module via UART.
            End-to-end test: sensor → FPGA → BLE → phone.

Week 13-14: Design custom PCB combining FPGA + sensors + BLE.
            Order from JLCPCB. 3D print enclosure.

Week 15-16: Power optimization: clock gating, duty cycling.
            Measure actual power consumption. Battery life test.

Week 17-18: Bench test with ECG simulator. Compare Verilog vs Python accuracy.
            Run PhysioNet replay through FPGA, log all decisions.

Week 19-20: Human pilot test. Signal quality assessment.
            Iterate on form factor and electrode placement.
```

## B10. Verification Strategy (Track B)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FPGA VERIFICATION FLOW                       │
│                                                                  │
│  1. UNIT TEST (per module)                                       │
│     Python generates test vectors → Verilog testbench            │
│     Compare output bit-exact against Python reference            │
│                                                                  │
│  2. INTEGRATION TEST                                             │
│     Replay UCI arrhythmia dataset through full pipeline           │
│     ecg_stimulus.v feeds features → capture decisions            │
│     python_checker.py compares vs Enhanced_model LOOCV results   │
│     PASS: accuracy within 1% of Python (fixed-point rounding)    │
│                                                                  │
│  3. HARDWARE-IN-THE-LOOP                                         │
│     ECG simulator → MAX86150 → FPGA → UART → PC                 │
│     Real analog signals, real sensor, verify timing              │
│                                                                  │
│  4. HUMAN VALIDATION                                             │
│     Wear device → log decisions + raw ECG                        │
│     Physician reviews flagged events                             │
└─────────────────────────────────────────────────────────────────┘
```

---

# ========================================================================
# SHARED: PHONE APP, VALIDATION & TESTING
# ========================================================================

## Phone App (both tracks)

```
┌────────────────────────────────────────┐
│         CDS Health Monitor App         │
├────────────────────────────────────────┤
│                                        │
│  Heart Rate:  72 bpm      [HEALTHY]    │
│  SpO2:       98%                       │
│  BP:         120/78 mmHg               │
│  Temp:       36.8 C                    │
│  Resp Rate:  16 /min                   │
│                                        │
│  ┌────────────────────────────────┐    │
│  │ ECG Waveform (real-time)       │    │
│  │ ~~~~~~~~~~~~~~~~~~~~           │    │
│  └────────────────────────────────┘    │
│                                        │
│  CDS Decision: HEALTHY                 │
│  Assurance Factor: 0.87                │
│  Model Used: CDS (confident)           │
│                                        │
│  [History]  [Settings]  [Share w/ Dr]  │
└────────────────────────────────────────┘
```

Build with: **Flutter** (cross-platform) or **React Native**.
BLE library: `flutter_blue_plus` or `react-native-ble-plx`.

## Validation Stages (both tracks)

| Stage | Data source | Pass criteria |
|-------|-------------|---------------|
| 1. Algorithm match | UCI arrhythmia (452 users) | Accuracy within 1% of Python LOOCV |
| 2. Signal chain | PhysioNet MIT-BIH | Feature extraction error < 5% |
| 3. Bench test | ECG simulator | Decision < 2s latency, < 10 mW |
| 4. Wearability | 10 healthy volunteers | Clean ECG > 90% of recording time |
| 5. Clinical pilot | 20 patients (mixed) | Sensitivity > 80%, Specificity > 90% |

---

# ========================================================================
# TRACK COMPARISON
# ========================================================================

| Factor | Track A: Embedded | Track B: FPGA |
|--------|-------------------|---------------|
| **Language** | C/C++ (Arduino) | Verilog/SystemVerilog |
| **Dev board** | Arduino Nano 33 BLE ($33) | Tang Nano 9K ($15) + BLE module ($5) |
| **Prototype BOM** | ~$70 | ~$66 |
| **Time to first demo** | ~4 weeks | ~6 weeks |
| **Processing latency** | ~1 ms (sequential) | ~128 us (parallel) |
| **Power consumption** | ~3.5 mA | ~2.2 mA |
| **Battery life (500mAh)** | ~6 days | ~9.5 days |
| **BLE** | Built-in | External module needed |
| **Debugging** | Serial print, GDB | Waveform viewer, ILA |
| **Scaling to production** | Easy (many ARM vendors) | Harder (FPGA NRE cost) |
| **Best for** | Quick prototype, software-first | Lowest power, research paper, FPGA learning |
| **Skills needed** | C/C++, Arduino, embedded | Verilog, digital design, timing analysis |

**Recommendation**: Start with Track A to validate the algorithm on real
signals quickly. Then port the classifier core to Track B for power
optimization and to demonstrate FPGA feasibility in your thesis/paper.

---

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Single-lead ECG fewer features than 12-lead UCI | Reduced accuracy | Retrain CDS on single-lead features + PPG features |
| Motion artifacts during activity | False alarms | IMU-based rejection; classify only in low-motion windows |
| CDS ranges from UCI don't generalize | Poor real-world accuracy | Per-patient calibration period; collect new dataset |
| FPGA too small for ESN reservoir | Can't fit Track B | Reduce to 50 reservoir nodes; or ANN-only on FPGA |
| BLE latency delays alerts | Late alarm | Process entirely on-device; BLE only for display |
| Fixed-point rounding degrades accuracy | Track B mismatch | Validate Q8.8 vs float32 on full UCI dataset first |

---

## References

- Alam et al., "A Wearable Tele-Health System towards Monitoring COVID-19
  and Chronic Diseases," IEEE RBME, 2021 (PMC8905615)
- PPG-Based BP Estimation Meta-Analysis, IEEE JBHI, 2024
- Efficient ECG Arrhythmia Detection on FPGA, ETASR 2025
- Resource-Constrained On-Chip AI Classifier, MDPI Electronics, 2025
- FPGA-based 1D-CNN Accelerator for Real-Time Arrhythmia, JRTIP 2025
