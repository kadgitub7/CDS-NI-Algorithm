/*
 * ESP32-S3 + AD8232 ECG Monitor — BLE Streaming Firmware
 * ======================================================
 *
 * Hardware wiring (AD8232 -> ESP32-S3):
 *   OUTPUT  -> GPIO 4  (ADC1_CH3, analog ECG signal)
 *   LO+     -> GPIO 5  (leads-off detection +)
 *   LO-     -> GPIO 6  (leads-off detection -)
 *   3.3V    -> 3V3
 *   GND     -> GND
 *   SDN     -> not connected (leave floating = active)
 *
 * ECG electrode placement (3-electrode):
 *   RA (Right Arm) -> right collarbone area
 *   LA (Left Arm)  -> left collarbone area
 *   RL (Right Leg)  -> lower right abdomen (reference/ground)
 *
 * BLE service:
 *   Service UUID:        0000ecg0-0000-1000-8000-00805f9b34fb
 *   ECG Data Char:       0000ecg1-0000-1000-8000-00805f9b34fb  (notify)
 *   Control Char:        0000ecg2-0000-1000-8000-00805f9b34fb  (write)
 *
 * Data format:
 *   Each BLE notification = 40 bytes = 20 samples x 2 bytes (uint16 LE)
 *   Sampling rate: 300 Hz  (notification every ~66.7ms)
 *   ADC resolution: 12-bit (0-4095)
 *
 * Control commands (write to control characteristic):
 *   "START"  -> begin streaming
 *   "STOP"   -> stop streaming
 *   "STATUS" -> reply with device status via notify
 */

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#define ECG_PIN       4
#define LEADS_OFF_P   5
#define LEADS_OFF_N   6

#define SAMPLE_RATE   300
#define SAMPLES_PER_PACKET  20
#define PACKET_BYTES  (SAMPLES_PER_PACKET * 2)

#define SERVICE_UUID        "0000ecg0-0000-1000-8000-00805f9b34fb"
#define ECG_CHAR_UUID       "0000ecg1-0000-1000-8000-00805f9b34fb"
#define CONTROL_CHAR_UUID   "0000ecg2-0000-1000-8000-00805f9b34fb"

BLEServer* pServer = NULL;
BLECharacteristic* pEcgChar = NULL;
BLECharacteristic* pControlChar = NULL;
bool deviceConnected = false;
bool streaming = false;

uint16_t sampleBuffer[SAMPLES_PER_PACKET];
int sampleIndex = 0;

hw_timer_t* sampleTimer = NULL;
volatile bool sampleReady = false;
volatile uint16_t latestSample = 0;
volatile bool leadsOff = false;

unsigned long lastStatusMs = 0;
unsigned long totalSamples = 0;

class ServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
        deviceConnected = true;
        Serial.println("Client connected");
    }

    void onDisconnect(BLEServer* pServer) {
        deviceConnected = false;
        streaming = false;
        Serial.println("Client disconnected");
        BLEDevice::startAdvertising();
    }
};

class ControlCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic* pChar) {
        std::string val = pChar->getValue();
        if (val == "START") {
            streaming = true;
            sampleIndex = 0;
            totalSamples = 0;
            Serial.println("Streaming started");
        } else if (val == "STOP") {
            streaming = false;
            Serial.println("Streaming stopped");
        } else if (val == "STATUS") {
            char status[64];
            snprintf(status, sizeof(status),
                     "OK rate=%d leads=%s samples=%lu",
                     SAMPLE_RATE,
                     leadsOff ? "OFF" : "ON",
                     totalSamples);
            pEcgChar->setValue((uint8_t*)status, strlen(status));
            pEcgChar->notify();
        }
    }
};

void IRAM_ATTR onSampleTimer() {
    leadsOff = (digitalRead(LEADS_OFF_P) == HIGH) ||
               (digitalRead(LEADS_OFF_N) == HIGH);
    if (leadsOff) {
        latestSample = 0;
    } else {
        latestSample = (uint16_t)analogRead(ECG_PIN);
    }
    sampleReady = true;
}

void setup() {
    Serial.begin(115200);
    Serial.println("CDS-ECG Sensor v1.0");

    pinMode(ECG_PIN, INPUT);
    pinMode(LEADS_OFF_P, INPUT);
    pinMode(LEADS_OFF_N, INPUT);

    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);

    BLEDevice::init("CDS-ECG-Sensor");
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());

    BLEService* pService = pServer->createService(SERVICE_UUID);

    pEcgChar = pService->createCharacteristic(
        ECG_CHAR_UUID,
        BLECharacteristic::PROPERTY_NOTIFY
    );
    pEcgChar->addDescriptor(new BLE2902());

    pControlChar = pService->createCharacteristic(
        CONTROL_CHAR_UUID,
        BLECharacteristic::PROPERTY_WRITE
    );
    pControlChar->setCallbacks(new ControlCallbacks());

    pService->start();

    BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->setMinPreferred(0x06);
    pAdvertising->setMinPreferred(0x12);
    BLEDevice::startAdvertising();

    Serial.println("BLE advertising started, waiting for connection...");

    // 300 Hz timer
    sampleTimer = timerBegin(0, 80, true);  // 80 MHz / 80 = 1 MHz tick
    timerAttachInterrupt(sampleTimer, &onSampleTimer, true);
    timerAlarmWrite(sampleTimer, 1000000 / SAMPLE_RATE, true);  // 3333 us
    timerAlarmEnable(sampleTimer);
}

void loop() {
    if (!sampleReady) return;
    sampleReady = false;

    if (!deviceConnected || !streaming) return;

    sampleBuffer[sampleIndex] = latestSample;
    sampleIndex++;
    totalSamples++;

    if (sampleIndex >= SAMPLES_PER_PACKET) {
        uint8_t packet[PACKET_BYTES];
        for (int i = 0; i < SAMPLES_PER_PACKET; i++) {
            packet[i * 2]     = sampleBuffer[i] & 0xFF;
            packet[i * 2 + 1] = (sampleBuffer[i] >> 8) & 0xFF;
        }
        pEcgChar->setValue(packet, PACKET_BYTES);
        pEcgChar->notify();
        sampleIndex = 0;
    }

    // Status print every 5 seconds
    unsigned long now = millis();
    if (now - lastStatusMs > 5000) {
        lastStatusMs = now;
        Serial.printf("Streaming: %lu samples, leads %s\n",
                      totalSamples, leadsOff ? "OFF" : "ON");
    }
}
