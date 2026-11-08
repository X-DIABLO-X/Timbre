import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from backend.config import detect_optimal_profile, resolve_media_path
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


@app.head("/api/video/dune")
async def head_dune_video():
    video_path = resolve_media_path("final_video.mp4")
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Dune video not found at {video_path}")
    file_size = video_path.stat().st_size
    return Response(
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Content-Type": "video/mp4",
        }
    )


@app.get("/api/video/dune")
async def stream_dune_video(request: Request):
    video_path = resolve_media_path("final_video.mp4")
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Dune video not found at {video_path}")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")

    if not range_header:
        def iterfile():
            with open(video_path, mode="rb") as f:
                while chunk := f.read(1024 * 1024):
                    yield chunk
        return StreamingResponse(
            iterfile(),
            media_type="video/mp4",
            headers={"Content-Length": str(file_size), "Accept-Ranges": "bytes"}
        )

    range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if not range_match:
        raise HTTPException(status_code=416, detail="Invalid Range header")

    start = int(range_match.group(1))
    end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
    end = min(end, file_size - 1)
    chunk_size = end - start + 1

    def iterfile_range(start_byte: int, end_byte: int):
        with open(video_path, mode="rb") as f:
            f.seek(start_byte)
            remaining = end_byte - start_byte + 1
            while remaining > 0:
                read_bytes = min(remaining, 512 * 1024)
                data = f.read(read_bytes)
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
    }
    return StreamingResponse(
        iterfile_range(start, end),
        status_code=206,
        media_type="video/mp4",
        headers=headers,
    )


@app.get("/api/captions/dune")
@app.get("/api/cwi/dune")
async def get_dune_captions():
    cwi_path = resolve_media_path("final_video.cwi.json")
    if not cwi_path.exists():
        from backend.ml_engine.dune_cwi_generator import convert_dune_srt_to_cwi
        convert_dune_srt_to_cwi()
    if not cwi_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "final_video.cwi.json not found", "checked_path": str(cwi_path)},
        )
    try:
        data = json.loads(cwi_path.read_text(encoding="utf-8"))
        return JSONResponse(content=data)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/process_audio")
async def process_uploaded_audio(file: UploadFile = File(...)) -> JSONResponse:
    try:
        pcm_bytes = await file.read()
        cfg = detect_optimal_profile()
        pipeline = ParallelAudioPipeline(config=cfg)
        spec = await pipeline.ingest_pcm_chunk(
            pcm_bytes=pcm_bytes,
            sample_rate=16000,
            is_float32=False,
            mode="buffered",
            locale="en",
        )
        if spec is None:
            return JSONResponse(content={"status": "silence", "words": []})
        return JSONResponse(content=spec.model_dump())
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})



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
