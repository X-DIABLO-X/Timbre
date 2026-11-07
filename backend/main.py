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
