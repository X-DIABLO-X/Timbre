# Wearable Neckband Companion Hardware Wiring Specification (`hardware_wiring.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint (Section 5.3 & 6.2 Tier 5)](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [Weave Spatial Voice Compass & Neckband (SNU, 2026)](https://xlaude2040.com/weave/), [Tactile Emotions Haptic Captioning (CHI 2025)](https://saadh.info/papers/pataca-chi-25/pataca-chi25.pdf)

---

## 1. System Overview

The wearable companion node provides 360° spatial awareness and discrete directional haptics for multi-speaker environments:
- **Compute:** Raspberry Pi Zero 2 W (running Raspberry Pi OS Lite 64-bit).
- **Microphone Array:** Seeed Studio ReSpeaker XVF3800 4-microphone circular array with onboard hardware acoustic front-end (DoA, AEC, beamforming).
- **Tactile Actuators:** Dual 10mm coin vibration ERM motors (Left and Right collarbone / mastoid positioning).
- **Host Link:** Low-latency Wi-Fi TCP streaming (`port 8766`) to host machine (`backend/wearable_bridge/tcp_compass_server.py`).

---

## 2. Raspberry Pi Zero 2 W 40-Pin GPIO Allocation

| Physical Pin | BCM / GPIO | Signal Name | Connected Component / Role | Electrical Specs |
| :---: | :---: | :---: | :---: | :---: |
| **Pin 1** | — | `3V3 Power` | Logic Pull-ups / Sensor logic (if needed) | 3.3V DC (max 50mA) |
| **Pin 2** | — | `5V Power` | Transistor Collector/Drain rail for coin motors | 5.0V DC (from USB input rail) |
| **Pin 6** | — | `GND` | Ground Reference | 0V Common Ground |
| **Pin 9** | — | `GND` | Motor Driver Ground Return | 0V Common Ground |
| **Pin 11** | `GPIO 17` | `HAPTIC_LEFT_PWM` | Left Coin Vibration Motor Driver Base/Gate | 3.3V LVCMOS PWM/GPIO |
| **Pin 13** | `GPIO 27` | `HAPTIC_RIGHT_PWM` | Right Coin Vibration Motor Driver Base/Gate | 3.3V LVCMOS PWM/GPIO |
| **Pin 14** | — | `GND` | Digital Ground | 0V Common Ground |
| **Micro-USB OTG**| — | `USB D+ / D-` | Seeed Studio ReSpeaker XVF3800 Array | USB 2.0 Full-Speed Host |

---

## 3. Coin Motor Driver Schematic (Low-Side NPN / N-Channel MOSFET)

Directly driving ERM coin motors from GPIO pins will destroy the SoC due to overcurrent (>16mA) and inductive kickback voltages (>20V). A low-side driver circuit must be used for both Left (`GPIO 17`) and Right (`GPIO 27`) channels.

```
       +5V Rail (Pin 2)
          │
          ├───[ + Motor Terminal ]
          │       (10mm Coin ERM)
          ├───[ - Motor Terminal ]───┬──────────────┐
          │                          │              │
        [ ▲ ] 1N4148 / 1N5819       │              │
        [ ──] Flyback Diode        │ (Collector)   │ (Drain)
          │   (Cathode to +5V)     [ c ] 2N2222    [ d ] 2N7000 / BSS138
          │                        b─[ b ] NPN     g─[ g ] N-MOSFET
          │                          [ e ] (Emitter)[ s ] (Source)
          │                            │              │
GPIO Pin ─┴──[ 1 kΩ Resistor ]─────────┘              │
(17 / 27)                                             │
                                                      │
GND Rail (Pin 6/9) ───────────────────────────────────┴───────────────
```

### Component Parts List:
1. **Transistor / Switch:**
   - Option A: **2N2222** / **2N3904** NPN BJT (with 1 kΩ base resistor).
   - Option B: **2N7000** or **BSS138** N-channel MOSFET (logic-level gate threshold $V_{GS(th)} \le 1.8\text{V}$, with 100 Ω gate resistor and 100 kΩ gate pull-down).
2. **Flyback Diode:** **1N4148** or **1N5819** Schottky diode placed antiparallel across motor terminals to clamp inductive flyback spikes.
3. **Decoupling Capacitor:** **10 µF** electrolytic + **0.1 µF** ceramic capacitor placed across +5V and GND near the motor connectors to suppress supply rail ripple caused by motor start-up transients.

---

## 4. ReSpeaker XVF3800 USB Interface

- **Interface:** Micro-USB to USB-C / Micro-USB OTG cable connecting Raspberry Pi Zero 2 W USB Data port to ReSpeaker XVF3800 USB port.
- **Power:** ReSpeaker draws ~220mA at 5V, supplied directly from the Pi's OTG port.
- **Vendor / Product ID:** `0x2886:0x001A` (Seeed Technology Co., Ltd. ReSpeaker XVF3800).
- **Control Interface:** Standard USB HID / vendor control transfers read DoA register `0x001A` reporting azimuth in degrees `[-180.0, +180.0]`.

---

## 5. Timing & Haptic Feedback Cadence

1. **DoA Polling:** Every **120 ms** (`8.33 Hz`), `wearable_pi_node/pi_neckband_daemon.py` polls the microphone array, applies circular averaging, and emits a `DOA_BEARING_UPDATE` JSON packet.
2. **Haptic Pulse Protocol:**
   - **Active Duration:** `100 ms` ON.
   - **Minimum Cycle Period:** `300 ms` total (`100 ms` ON, `200 ms` quiescent cooling period).
   - **Trigger Conditions:** Confirmed speaker attribution change OR emphasis prominence score $\ge 0.80$.
   - **Spatial Mapping:**
     - Azimuth $\le -15^\circ$: Left motor only (`GPIO 17`).
     - Azimuth $\ge +15^\circ$: Right motor only (`GPIO 27`).
     - Azimuth between $-15^\circ$ and $+15^\circ$: Both motors in unison (indicating straight ahead / front-facing speaker).
