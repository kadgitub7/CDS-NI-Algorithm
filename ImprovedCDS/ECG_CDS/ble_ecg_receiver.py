"""BLE ECG Receiver — connects to ESP32 CDS-ECG-Sensor and records ECG data.

Uses the `bleak` library for cross-platform BLE (Windows/Mac/Linux/Android via Termux).

Install:  pip install bleak

Usage:
    from ble_ecg_receiver import BleEcgReceiver

    receiver = BleEcgReceiver()
    await receiver.scan_and_connect()
    signal = await receiver.record(duration_sec=30)
    await receiver.disconnect()
"""
import asyncio
import struct
import time
import numpy as np
from typing import Optional

try:
    from bleak import BleakClient, BleakScanner
    HAS_BLEAK = True
except ImportError:
    HAS_BLEAK = False

SERVICE_UUID = "0000ecg0-0000-1000-8000-00805f9b34fb"
ECG_CHAR_UUID = "0000ecg1-0000-1000-8000-00805f9b34fb"
CONTROL_CHAR_UUID = "0000ecg2-0000-1000-8000-00805f9b34fb"

SAMPLE_RATE = 300
SAMPLES_PER_PACKET = 20
ADC_MAX = 4095
ADC_VREF = 3.3


class BleEcgReceiver:
    """Connects to ESP32 CDS-ECG-Sensor via BLE and records raw ECG."""

    def __init__(self, device_name: str = "CDS-ECG-Sensor"):
        if not HAS_BLEAK:
            raise ImportError("Install bleak: pip install bleak")
        self.device_name = device_name
        self.client: Optional[BleakClient] = None
        self._samples: list = []
        self._recording = False
        self._leads_off_count = 0

    def _notification_handler(self, sender, data: bytearray):
        """Parse BLE notification: 20 samples x 2 bytes (uint16 LE)."""
        if not self._recording:
            return
        n_samples = len(data) // 2
        for i in range(n_samples):
            raw = struct.unpack_from("<H", data, i * 2)[0]
            if raw == 0:
                self._leads_off_count += 1
            voltage = (raw / ADC_MAX) * ADC_VREF
            self._samples.append(voltage)

    async def scan(self, timeout: float = 10.0) -> Optional[str]:
        """Scan for CDS-ECG-Sensor. Returns device address or None."""
        print(f"Scanning for '{self.device_name}'...")
        devices = await BleakScanner.discover(timeout=timeout)
        for d in devices:
            if d.name and self.device_name in d.name:
                print(f"  Found: {d.name} [{d.address}] RSSI={d.rssi}")
                return d.address
        print("  Device not found.")
        return None

    async def connect(self, address: str) -> bool:
        """Connect to a known device address."""
        print(f"Connecting to {address}...")
        self.client = BleakClient(address)
        connected = await self.client.connect()
        if connected:
            print("  Connected.")
            await self.client.start_notify(ECG_CHAR_UUID, self._notification_handler)
        return connected

    async def scan_and_connect(self, timeout: float = 10.0) -> bool:
        """Scan for device and connect."""
        address = await self.scan(timeout)
        if address is None:
            return False
        return await self.connect(address)

    async def start_streaming(self):
        """Send START command to begin ECG data streaming."""
        if self.client and self.client.is_connected:
            await self.client.write_gatt_char(CONTROL_CHAR_UUID, b"START")
            print("Streaming started.")

    async def stop_streaming(self):
        """Send STOP command."""
        if self.client and self.client.is_connected:
            await self.client.write_gatt_char(CONTROL_CHAR_UUID, b"STOP")
            print("Streaming stopped.")

    async def record(self, duration_sec: float = 30.0) -> np.ndarray:
        """Record ECG for the specified duration. Returns raw signal array."""
        self._samples = []
        self._recording = True
        self._leads_off_count = 0

        await self.start_streaming()

        expected_samples = int(duration_sec * SAMPLE_RATE)
        print(f"Recording {duration_sec}s ({expected_samples} samples at {SAMPLE_RATE}Hz)...")

        t0 = time.time()
        while len(self._samples) < expected_samples:
            elapsed = time.time() - t0
            if elapsed > duration_sec + 5:
                print(f"  Timeout after {elapsed:.1f}s with {len(self._samples)} samples")
                break
            await asyncio.sleep(0.1)

        await self.stop_streaming()
        self._recording = False

        signal = np.array(self._samples[:expected_samples], dtype=float)
        actual_duration = len(signal) / SAMPLE_RATE

        print(f"  Recorded {len(signal)} samples ({actual_duration:.1f}s)")
        if self._leads_off_count > 0:
            pct = 100 * self._leads_off_count / max(len(signal), 1)
            print(f"  WARNING: {self._leads_off_count} leads-off samples ({pct:.1f}%)")

        return signal

    async def disconnect(self):
        """Disconnect from BLE device."""
        if self.client and self.client.is_connected:
            await self.client.stop_notify(ECG_CHAR_UUID)
            await self.client.disconnect()
            print("Disconnected.")

    async def get_status(self) -> str:
        """Query device status."""
        if self.client and self.client.is_connected:
            await self.client.write_gatt_char(CONTROL_CHAR_UUID, b"STATUS")
            await asyncio.sleep(0.5)
        return "Status requested (check notify)"


class SimulatedReceiver:
    """Simulated receiver for testing without hardware.

    Generates synthetic ECG with optional AF patterns.
    """

    def __init__(self, mode: str = "normal"):
        self.mode = mode
        self.sample_rate = SAMPLE_RATE

    async def scan_and_connect(self, timeout=10.0) -> bool:
        print("[SIM] Simulated ECG device connected")
        return True

    async def record(self, duration_sec: float = 30.0) -> np.ndarray:
        """Generate synthetic ECG signal."""
        n = int(duration_sec * self.sample_rate)
        t = np.arange(n) / self.sample_rate

        ecg = np.zeros(n)

        if self.mode == "normal":
            rr_mean = 0.8
            rr_std = 0.02
        elif self.mode == "afib":
            rr_mean = 0.7
            rr_std = 0.2
        else:
            rr_mean = 0.85
            rr_std = 0.05

        rng = np.random.RandomState(42)
        beat_time = 0.3
        beat_times = []
        while beat_time < duration_sec - 0.5:
            beat_times.append(beat_time)
            rr = max(0.3, rng.normal(rr_mean, rr_std))
            beat_time += rr

        for bt in beat_times:
            ecg += 1.5 * np.exp(-0.5 * ((t - bt) / 0.02) ** 2)
            ecg += 0.3 * np.exp(-0.5 * ((t - bt + 0.16) / 0.04) ** 2)
            ecg += 0.4 * np.exp(-0.5 * ((t - bt - 0.2) / 0.06) ** 2)
            ecg -= 0.2 * np.exp(-0.5 * ((t - bt - 0.04) / 0.015) ** 2)

        ecg += 0.05 * rng.randn(n)
        # Scale to ADC voltage range
        ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min()) * ADC_VREF

        print(f"[SIM] Generated {self.mode} ECG: {n} samples, {duration_sec}s")
        return ecg

    async def disconnect(self):
        print("[SIM] Disconnected")


if __name__ == "__main__":
    async def main():
        # Demo with simulated receiver
        receiver = SimulatedReceiver(mode="normal")
        await receiver.scan_and_connect()
        signal = await receiver.record(duration_sec=30)
        await receiver.disconnect()
        print(f"Signal shape: {signal.shape}, range: [{signal.min():.3f}, {signal.max():.3f}]")

    asyncio.run(main())
