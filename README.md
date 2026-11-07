# Automated "Caption with Intention" (CWI) Platform (`v2.0-cwi`)

Production-grade, real-time and offline expressive subtitle platform modeled after FCB Chicago and the Chicago Hearing Society's [Caption with Intention©](https://www.captionwithintention.org/), with dual-clock read-ahead typography, 5 kinetic animation channels, and multi-speaker attribution.

---

## 1. System Architecture

The platform is decoupled into two primary systems connected by the deterministic `CaptionSpec v2.0-cwi` contract:

- **Part 1: Parallel ML Audio Engine (`backend/`)**
  - **Stage 0 (`stage0_vad_separator.py`):** 16kHz mono float32 normalization, Silero VAD gate, and optional SCNet/Demucs vocal separation.
  - **Stream 1 (`stream1_asr.py`):** Verbatim ASR with sub-50ms word timestamps via CrisperWhisper 2.0 (CTranslate2 speculative decoding) or Nemotron + Parakeet TDT v3.
  - **Stream 2 (`stream2_diarization.py`):** Streaming Sortformer 4spk v2.1 with 4-state attribution (`unknown` -> `provisional` -> `stable` -> `corrected`) and 512MB RAM online cosine clustering fallback.
  - **Stream 3 (`stream3_prosody_stress.py`):** 75ms/20ms $F_0$ & RMS extraction, 300ms pooling, running speaker median $F_0$ tracking, and WhiStress Layer-9 prominence head.
  - **Stream 4 (`stream4_paralinguistics.py`):** SenseVoice-Small/Large audio event detection (`<|BGM|>`, `[laughter]`, `[sighs]`) and emotion2vec+ 50Hz frame affect.
  - **Joiner & Orchestrator (`joiner.py`, `orchestrator.py`):** Concurrently dispatches streams via `asyncio.gather` and builds unified `CaptionSpec` frames.
  - **FastAPI Backend (`main.py`):** WebSocket (`ws://127.0.0.1:8765/ws`) and SSE endpoints.

- **Part 2: Deterministic Renderer & Client (`extension/` & `frontend_renderer/`)**
  - **Dual-Clock Controller:** `-1.75s` read-ahead presentation offset with `420ms` exposure floor.
  - **Single-Delay CSS Scheduler:** Binds all 5 kinetic animation channels in a single CSS pass on the GPU compositor.
  - **5 Kinetic Channels:**
    1. *Colour Turn:* Horizontal `background-clip: text` gradient wipe from word onset to offset.
    2. *Pop:* Universal 1.15x scale impulse at word onset $t_0$.
    3. *Crest:* 0.72x to 1.62x volume-driven scale settling to 0.70 of peak.
    4. *Weight:* 80 Hz ($wght 900$) to 250 Hz ($wght 340$) anchored to speaker median $F_0$ (with Safari $wght 404$ fix).
    5. *Hold Lift:* -0.525em vertical lift when flanked by $\ge 250ms$ silence.
  - **Chrome Manifest V3 Extension (`extension/`):** Real-time tab audio capture with mandatory speaker passthrough, AudioWorklet 16kHz resampler, and in-browser WebGPU worker (Moonshine Base).
  - **Offline ASS Burn-In Engine (`frontend_renderer/offline_burnin/`):** Compiles `CaptionSpec` to Advanced SubStation Alpha (`.ass`) with `\kf` karaoke tags and burns in via FFmpeg `libass`.

---

## 2. Monorepo Directory Structure

```text
├── pyproject.toml                     # Python dependencies & pytest configuration
├── package.json                       # Node.js workspace configuration
├── shared_schemas/                    # JSON schema & speaker palette contracts
│   ├── caption_spec.schema.json
│   ├── speaker_palette.json
│   └── roboto_flex_axes.json
├── backend/                           # FastAPI ML & Prosody Engine
│   ├── config.py                      # Hardware auto-detection
│   ├── main.py                        # FastAPI WebSocket server
│   ├── schemas/                       # Pydantic v2 CaptionSpec models
│   ├── ml_engine/                     # Stage 0 + Streams 1-4 + Orchestrator
│   ├── native_host/                   # Chrome Native Messaging launcher
│   └── wearable_bridge/               # TCP server for spatial compass
├── training/                          # Custom heads & ONNX export
│   ├── models/                        # WhiStress Layer-9 & Emotion2Vec heads
│   ├── datasets/                      # TinyStress, BU-RNC, VocalSound loaders
│   └── export_onnx_hybrid.py          # WebGPU hybrid quantization
├── extension/                         # Chrome MV3 Extension
│   ├── manifest.json
│   ├── background.js
│   ├── offscreen/                     # Tab capture & AudioWorklet resampler
│   ├── workers/                       # In-browser WebGPU worker
│   └── content/                       # Shadow DOM overlay injector
├── frontend_renderer/                 # DOM Renderer, React 19 Player & ASS Compiler
│   ├── src/core/                      # CwiRenderer, DualClock, CSS scheduler
│   ├── src/player/                    # React 19 VOD/Live Studio
│   └── offline_burnin/                # ASS compiler & FFmpeg burn-in wrapper
├── wearable_pi_node/                  # Raspberry Pi Zero 2 W wearable companion
└── tests/                             # 256 deterministic verification invariant tests
```

---

## 3. Quick Start

### Python ML Backend
```bash
pip install -e .
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
```

### Verification Test Suite
```bash
pytest tests/test_all_256_invariants.py -v
```

### Chrome Extension
1. Open `chrome://extensions/`
2. Enable **Developer mode**
3. Click **Load unpacked** and select the `extension/` directory.

### React Player & Offline Burn-In
```bash
npm install
npm run dev:player
```
