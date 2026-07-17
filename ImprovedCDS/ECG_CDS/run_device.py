"""CDS-ECG Edge Device — Main Entry Point

Ties together the full pipeline:
  1. Connect to ESP32+AD8232 sensor via BLE (or use simulator)
  2. Record ECG for a configurable duration
  3. Extract 188 features from raw signal
  4. Run CDS-OVR classification
  5. Display result with confidence details

Usage:
    # With real hardware:
    python run_device.py

    # With simulated ECG (no hardware needed):
    python run_device.py --sim normal
    python run_device.py --sim afib

    # Custom recording duration:
    python run_device.py --duration 45

    # Classify a saved .npy signal file:
    python run_device.py --file recording.npy

Requirements:
    pip install numpy scipy bleak
    (sklearn optional, used by some feature extractors)
"""
import argparse
import asyncio
import json
import sys
import time
import numpy as np
from pathlib import Path

from edge_classifier import load_model
from ble_ecg_receiver import BleEcgReceiver, SimulatedReceiver, SAMPLE_RATE


MODEL_PATH = Path(__file__).parent / "cds_model.json"
RECORDING_DIR = Path(__file__).parent / "recordings"


def print_result(pred_cls, pred_name, details, duration):
    """Display classification result."""
    print("\n" + "=" * 50)
    print("  CDS-ECG Classification Result")
    print("=" * 50)

    if pred_cls == 1:
        print(f"\n  Result:  NORMAL SINUS RHYTHM")
    else:
        print(f"\n  Result:  ABNORMAL (possible arrhythmia)")
        print(f"  Action:  Consult a physician for further evaluation")

    print(f"\n  Recording duration: {duration:.1f}s")
    print(f"  Classification time: {details.get('classify_time_ms', 0):.0f}ms")

    print(f"\n  Scores:")
    for cls_key, d in details["class_details"].items():
        name = "Normal" if cls_key == "1" else "Abnormal"
        print(f"    {name:10s}: ratio={d['ratio']:.3f} "
              f"(for={d['af_for']:.3f}, against={d['af_against']:.3f}, "
              f"feats={d['n_features_used']})")
    print(f"  Threshold: {details['effective_threshold']:.3f}")

    print("\n  NOTE: This is a screening tool, not a clinical diagnosis.")
    print("  Always consult a healthcare professional.")
    print("=" * 50)


async def run_live(args):
    """Run with live BLE sensor."""
    if not MODEL_PATH.exists():
        print(f"Model file not found: {MODEL_PATH}")
        print("Run 'python model_export.py' first to train and export the model.")
        return

    model = load_model(str(MODEL_PATH))
    print(f"Loaded CDS model ({model.n_features} features)")

    receiver = BleEcgReceiver()
    connected = await receiver.scan_and_connect(timeout=15.0)
    if not connected:
        print("\nCould not find CDS-ECG-Sensor.")
        print("Make sure:")
        print("  1. ESP32 is powered on and running esp32_ad8232_ble.ino")
        print("  2. Bluetooth is enabled on this device")
        print("  3. ECG electrodes are attached")
        return

    try:
        while True:
            input("\nPress Enter to start recording (Ctrl+C to quit)...")
            signal = await receiver.record(duration_sec=args.duration)

            if len(signal) < SAMPLE_RATE * 5:
                print("Recording too short (need at least 5 seconds). Try again.")
                continue

            RECORDING_DIR.mkdir(exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            rec_path = RECORDING_DIR / f"ecg_{ts}.npy"
            np.save(rec_path, signal)
            print(f"  Saved recording to {rec_path}")

            print("Classifying...")
            t0 = time.time()
            pred_cls, pred_name, details = model.predict_from_signal(signal, SAMPLE_RATE)
            details["classify_time_ms"] = (time.time() - t0) * 1000

            print_result(pred_cls, pred_name, details, len(signal) / SAMPLE_RATE)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await receiver.disconnect()


async def run_simulated(args):
    """Run with simulated ECG."""
    if not MODEL_PATH.exists():
        print(f"Model file not found: {MODEL_PATH}")
        print("Run 'python model_export.py' first to train and export the model.")
        return

    model = load_model(str(MODEL_PATH))
    print(f"Loaded CDS model ({model.n_features} features)")

    receiver = SimulatedReceiver(mode=args.sim)
    await receiver.scan_and_connect()
    signal = await receiver.record(duration_sec=args.duration)
    await receiver.disconnect()

    print("Classifying...")
    t0 = time.time()
    pred_cls, pred_name, details = model.predict_from_signal(signal, SAMPLE_RATE)
    details["classify_time_ms"] = (time.time() - t0) * 1000

    print_result(pred_cls, pred_name, details, len(signal) / SAMPLE_RATE)


async def run_from_file(args):
    """Classify a saved recording."""
    if not MODEL_PATH.exists():
        print(f"Model file not found: {MODEL_PATH}")
        print("Run 'python model_export.py' first to train and export the model.")
        return

    model = load_model(str(MODEL_PATH))
    print(f"Loaded CDS model ({model.n_features} features)")

    fpath = Path(args.file)
    if not fpath.exists():
        print(f"File not found: {fpath}")
        return

    if fpath.suffix == ".npy":
        signal = np.load(fpath)
    elif fpath.suffix == ".mat":
        import scipy.io
        d = scipy.io.loadmat(str(fpath))
        signal = d["val"].flatten().astype(float)
    elif fpath.suffix in (".csv", ".txt"):
        signal = np.loadtxt(str(fpath))
    else:
        print(f"Unsupported file format: {fpath.suffix}")
        print("Supported: .npy, .mat, .csv, .txt")
        return

    print(f"Loaded {fpath.name}: {len(signal)} samples ({len(signal)/SAMPLE_RATE:.1f}s)")

    print("Classifying...")
    t0 = time.time()
    pred_cls, pred_name, details = model.predict_from_signal(signal, SAMPLE_RATE)
    details["classify_time_ms"] = (time.time() - t0) * 1000

    print_result(pred_cls, pred_name, details, len(signal) / SAMPLE_RATE)


def main():
    parser = argparse.ArgumentParser(
        description="CDS-ECG Edge Device: Record and classify ECG"
    )
    parser.add_argument("--sim", type=str, default=None,
                        choices=["normal", "afib", "other"],
                        help="Use simulated ECG instead of real sensor")
    parser.add_argument("--file", type=str, default=None,
                        help="Classify a saved recording (.npy, .mat, .csv)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Recording duration in seconds (default: 30)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to exported model JSON")

    args = parser.parse_args()

    global MODEL_PATH
    if args.model:
        MODEL_PATH = Path(args.model)

    if args.file:
        asyncio.run(run_from_file(args))
    elif args.sim:
        asyncio.run(run_simulated(args))
    else:
        asyncio.run(run_live(args))


if __name__ == "__main__":
    main()
