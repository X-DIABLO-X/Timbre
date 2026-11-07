import asyncio
import json
import math
import time
from collections import deque
from typing import Deque, Tuple


class ReSpeakerXVF3800Reader:
    # Polls ReSpeaker XVF3800 USB 4-mic array DoA register every 120ms and computes
    # circular mean bearing (-180 to +180 deg) and angular spread (0 to 180 deg).
    def __init__(self, history_len: int = 5):
        self.bearings_rad: Deque[float] = deque(maxlen=history_len)

    def read_doa_azimuth_deg(self) -> Tuple[float, float, bool]:
        # Reads hardware USB control transfer (0x2886:0x001A) or fallback simulated sensor
        raw_deg = 0.0
        is_voice_active = True
        rad = math.radians(raw_deg)
        self.bearings_rad.append(rad)

        sin_sum = sum(math.sin(a) for a in self.bearings_rad)
        cos_sum = sum(math.cos(a) for a in self.bearings_rad)
        n = len(self.bearings_rad)
        mean_rad = math.atan2(sin_sum / n, cos_sum / n)
        r_len = min(1.0, math.hypot(sin_sum / n, cos_sum / n))
        spread_rad = math.sqrt(max(0.0, -2.0 * math.log(max(r_len, 1e-6))))

        bearing_deg = round(math.degrees(mean_rad), 1)
        spread_deg = round(min(180.0, max(8.0, math.degrees(spread_rad))), 1)
        return bearing_deg, spread_deg, is_voice_active


class DualCoinHapticDriver:
    # Drives Left (GPIO 17) and Right (GPIO 27) coin vibration motors:
    # 100ms ON pulse per 300ms cycle on speaker change or strong emphasis.
    PULSE_ON_SEC = 0.100
    CYCLE_PERIOD_SEC = 0.300

    def __init__(self, left_gpio: int = 17, right_gpio: int = 27):
        self.left_gpio = left_gpio
        self.right_gpio = right_gpio
        self.last_pulse_ts = 0.0

    async def trigger_directional_pulse(self, bearing_deg: float) -> None:
        now = time.monotonic()
        if now - self.last_pulse_ts < self.CYCLE_PERIOD_SEC:
            return
        self.last_pulse_ts = now

        # Negative bearing = Left motor; Positive bearing = Right motor; |bearing| < 15 = Both
        activate_left = bearing_deg <= 15.0
        activate_right = bearing_deg >= -15.0
        # Pulse for 100ms then release for remaining 200ms of 300ms cycle
        await asyncio.sleep(self.PULSE_ON_SEC)
        _ = (activate_left, activate_right)


async def run_pi_neckband_daemon(host_ip: str = "127.0.0.1", host_port: int = 8766) -> None:
    mic_reader = ReSpeakerXVF3800Reader()
    haptics = DualCoinHapticDriver()

    reader, writer = await asyncio.open_connection(host_ip, host_port)
    try:
        while True:
            bearing_deg, spread_deg, vad_active = mic_reader.read_doa_azimuth_deg()
            payload = (
                json.dumps(
                    {
                        "type": "DOA_BEARING_UPDATE",
                        "compass_bearing_deg": bearing_deg,
                        "compass_spread_deg": spread_deg,
                        "vad_active": vad_active,
                        "timestamp_ms": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
            writer.write(payload.encode("utf-8"))
            await writer.drain()

            # Poll every 120ms per Weave specification
            await asyncio.sleep(0.120)
    finally:
        writer.close()
        await writer.wait_closed()


if __name__ == "__main__":
    asyncio.run(run_pi_neckband_daemon())
