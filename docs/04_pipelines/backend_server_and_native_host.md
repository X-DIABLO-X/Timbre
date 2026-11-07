# FastAPI WebSocket/SSE Server, Chrome Native Host Launcher & TCP Compass Receiver (`backend_server_and_native_host.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint (Section 5.1 & 7)](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [kami-subs Native Host & WebSocket Server](https://github.com/MohammdKopa/kami-subs), [Weave Wearable TCP Bridge](https://xlaude2040.com/weave/)

---

## 1. Complete FastAPI WebSocket & SSE Server — `backend/main.py`

Write the following complete code to `backend/main.py`:

```python
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.config import detect_optimal_profile
from backend.ml_engine.orchestrator import ParallelAudioPipeline
from backend.wearable_bridge.tcp_compass_server import WearableCompassTCPServer

compass_server: Optional[WearableCompassTCPServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global compass_server
    cfg = detect_optimal_profile()
    compass_server = WearableCompassTCPServer(host="0.0.0.0", port=cfg.wearable_tcp_port)
    await compass_server.start()
    yield
    if compass_server:
        await compass_server.stop()


app = FastAPI(
    title="Automated Caption with Intention (CWI) ML Engine",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    cfg = detect_optimal_profile()
    return {
        "status": "ok",
        "schema_version": "2.0-cwi",
        "hardware_tier": cfg.tier,
        "device": cfg.device,
    }


@app.websocket("/ws")
async def websocket_caption_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    cfg = detect_optimal_profile()
    pipeline = ParallelAudioPipeline(config=cfg)

    sample_rate = 16000
    is_float32 = False
    mode = "live_streaming"
    locale = "en"

    try:
        while True:
            msg = await ws.receive()
            if "text" in msg and msg["text"]:
                ctrl = json.loads(msg["text"])
                if ctrl.get("type") == "SESSION_START":
                    sample_rate = int(ctrl.get("sample_rate", 16000))
                    is_float32 = ctrl.get("format") == "pcm_f32le"
                    mode = str(ctrl.get("mode", "live_streaming"))
                    locale = str(ctrl.get("locale", "en"))
            elif "bytes" in msg and msg["bytes"]:
                pcm_bytes = msg["bytes"]
                compass_state = compass_server.latest_state if compass_server else None
                spec = await pipeline.ingest_pcm_chunk(
                    pcm_bytes=pcm_bytes,
                    sample_rate=sample_rate,
                    is_float32=is_float32,
                    mode=mode,
                    locale=locale,
                    compass_state=compass_state,
                )
                if spec is not None:
                    await ws.send_text(spec.model_dump_json())
    except WebSocketDisconnect:
        pass
```

---

## 2. Chrome Native Messaging Auto-Launcher — `backend/native_host/launcher.py`

Write the following complete code to `backend/native_host/launcher.py` (and mark executable `chmod +x`):

```python
#!/usr/bin/env python3
import json
import socket
import struct
import subprocess
import sys
from pathlib import Path


def is_port_in_use(host: str = "127.0.0.1", port: int = 8765) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0


def read_native_message() -> dict | None:
    raw_len = sys.stdin.buffer.read(4)
    if len(raw_len) < 4:
        return None
    msg_len = struct.unpack("@I", raw_len)[0]
    payload = sys.stdin.buffer.read(msg_len).decode("utf-8")
    return json.loads(payload)


def send_native_message(obj: dict) -> None:
    encoded = json.dumps(obj).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("@I", len(encoded)))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    while True:
        msg = read_native_message()
        if msg is None:
            break
        if msg.get("command") == "START_FASTAPI_SERVER":
            port = int(msg.get("port", 8765))
            if not is_port_in_use("127.0.0.1", port):
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "backend.main:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                    ],
                    cwd=str(repo_root),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            send_native_message({"status": "running", "port": port})


if __name__ == "__main__":
    main()
```

---

## 3. Wearable Compass Async TCP Server — `backend/wearable_bridge/tcp_compass_server.py`

Write the following complete code to `backend/wearable_bridge/tcp_compass_server.py`:

```python
import asyncio
import json
from typing import Dict, Optional


class WearableCompassTCPServer:
    # Receives 120ms DoA bearing updates over TCP from the Raspberry Pi Zero 2 W neckband.
    def __init__(self, host: str = "0.0.0.0", port: int = 8766):
        self.host = host
        self.port = port
        self.server: Optional[asyncio.AbstractServer] = None
        self.latest_state: Optional[Dict[str, float]] = None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                data = json.loads(line.decode("utf-8"))
                if data.get("type") == "DOA_BEARING_UPDATE":
                    self.latest_state = {
                        "bearing_deg": float(data.get("compass_bearing_deg", 0.0)),
                        "spread_deg": float(data.get("compass_spread_deg", 12.0)),
                    }
        except Exception:
            pass
        finally:
            writer.close()

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle_client, self.host, self.port)

    async def stop(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()
```
