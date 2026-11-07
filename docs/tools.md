# Complete Software Stack, Dependencies & Environment Manifests (`tools.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper), [NVIDIA NeMo Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1), [WhiStress](https://arxiv.org/html/2505.19103v1), [Transformers.js WebGPU](https://maddevs.io/writeups/running-ai-models-locally-in-the-browser/)

---

## 1. Complete Tooling & Library Matrix by Subsystem

| Subsystem | Runtime / Language | Core Libraries & Exact Roles |
| :--- | :--- | :--- |
| **Part 1A: Python FastAPI ML Backend (`backend/`)** | Python `3.10`–`3.12` | `fastapi`, `uvicorn[standard]`, `websockets`, `pydantic>=2.7`, `numpy`, `scipy`, `librosa` / `pyworld` / `torchcrepe` (DSP $F_0$ & RMS), `silero-vad`, `faster-whisper` / `ctranslate2>=4.4.0` (CrisperWhisper 2.0 speculative decoding), `nemo_toolkit[asr]` (Streaming Sortformer v2.1, Nemotron, Parakeet TDT v3), `modelscope` / `funasr` (SenseVoice-Small/Large & emotion2vec+ & ERes2Net/CAM++), `pyannote.audio` (optional streaming diarization), `demucs` / `scnet` (optional VOD vocal stem separation). |
| **Part 1B: Apple Silicon Native Acceleration (`backend/`)** | macOS (`arm64`) | `pywhispercpp` / `whisper.cpp` (compiled with `WHISPER_COREML=1` for ANE encoder acceleration), `mlx`, `mlx-audio` (`mlx-community/emotion2vec-plus-large-mlx`), `coremltools` (Core ML Sortformer). |
| **Part 1C: Custom Training & ONNX Export (`training/`)** | Python `3.10+` + PyTorch `>=2.2` | `torch>=2.2.0`, `torchaudio`, `transformers>=4.44.0`, `datasets`, `accelerate`, `scikit-learn` (F1 evaluation), `onnx`, `onnxruntime`, `onnxruntime-tools` / `optimum` (Hybrid `fp32` encoder + `q4f16`/`q8` decoder quantization). |
| **Part 2A: Chrome MV3 Extension & WebGPU (`extension/`)** | Chrome/Edge `113+`, ES2022 | `@huggingface/transformers` (`>=3.1.0`, ONNX Runtime Web with `device: "webgpu"` and `"wasm"` SIMD fallback), `onnxruntime-web`, Chrome Extension APIs (`chrome.tabCapture`, `chrome.offscreen`, `chrome.runtime.connectNative`, `chrome.storage`), Web Audio API (`AudioContext`, `AudioWorkletNode`). |
| **Part 2B: Frontend Player & Offline `.ass` Burn-In (`frontend_renderer/`)** | TypeScript `5.5+`, React `19`, Python `3.10+`, FFmpeg `6.1+` | `react@19`, `react-dom@19`, `vite`, Shadow DOM (`attachShadow({ mode: 'open' })`), CSS Custom Properties & `@keyframes`, `ffmpeg` compiled with `--enable-libass --enable-libfreetype --enable-libharfbuzz`. |
| **Optional Wearable Companion (`wearable_pi_node/`)** | Raspberry Pi OS Lite (`arm64`/`armhf`) | `pyusb`, `pyserial`, `RPi.GPIO` / `gpiozero` (coin motor PWM), `asyncio` TCP client. |

---

## 2. Exact `pyproject.toml` — Write to `cwi-platform/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cwi-platform"
version = "2.0.0"
description = "Automated Caption with Intention Real-Time & Offline Expressive Subtitle Platform"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.6",
    "websockets>=13.0",
    "pydantic>=2.8.2",
    "numpy>=1.26.4,<2.0.0",
    "scipy>=1.13.1",
    "librosa>=0.10.2.post1",
    "soundfile>=0.12.1",
    "torch>=2.3.0",
    "torchaudio>=2.3.0",
    "transformers>=4.44.2",
    "ctranslate2>=4.4.0",
    "faster-whisper>=1.0.3",
    "silero-vad>=5.1",
    "funasr>=1.1.9",
    "modelscope>=1.18.0",
    "scikit-learn>=1.5.1",
    "pyyaml>=6.0.2",
    "onnx>=1.16.2",
    "onnxruntime>=1.19.0",
]

[project.optional-dependencies]
nemo = [
    "nemo_toolkit[asr]>=2.0.0",
    "hydra-core>=1.3.2",
    "pytorch-lightning>=2.3.0",
]
apple_silicon = [
    "mlx>=0.18.0",
    "coremltools>=8.0",
    "pywhispercpp>=1.2.0",
]
separation = [
    "demucs>=4.0.1",
]
dev = [
    "pytest>=8.3.2",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.27.2",
    "jsonschema>=4.23.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
```

---

## 3. Exact `package.json` — Write to `cwi-platform/package.json`

```json
{
  "name": "cwi-platform-web",
  "version": "2.0.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev:player": "vite --config frontend_renderer/vite.config.ts",
    "build:player": "tsc && vite build --config frontend_renderer/vite.config.ts",
    "build:extension": "node scripts/build_extension.mjs",
    "test:unit": "vitest run"
  },
  "dependencies": {
    "@huggingface/transformers": "^3.1.2",
    "onnxruntime-web": "^1.19.2",
    "react": "^19.0.0",
    "react-dom": "^19.0.0"
  },
  "devDependencies": {
    "@types/chrome": "^0.0.270",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^4.3.2",
    "typescript": "^5.6.2",
    "vite": "^5.4.8",
    "vitest": "^2.1.2"
  }
}
```

---

## 4. Chrome Native Messaging Host Manifest (`backend/native_host/com.cwi.engine.json`)

To allow `extension/background.js` to launch the Python FastAPI server automatically when the user clicks **Start** (without opening a terminal), register this Native Messaging Host manifest ([kami-subs](https://github.com/MohammdKopa/kami-subs), [Gemive](https://github.com/letr1n1ty/Gemive)):

```json
{
  "name": "com.cwi.engine",
  "description": "Automated Caption with Intention Local Python FastAPI ML Launcher",
  "path": "/ABSOLUTE/PATH/TO/cwi-platform/backend/native_host/launcher.py",
  "type": "stdio",
  "allowed_origins": [
    "chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/"
  ]
}
```

### OS Registration Paths for `com.cwi.engine.json`:
- **Linux:** `~/.config/google-chrome/NativeMessagingHosts/com.cwi.engine.json`
- **macOS:** `~/Library/Application Support/Google/Chrome/NativeMessagingHosts/com.cwi.engine.json`
- **Windows:** Registry key `HKCU\Software\Google\Chrome\NativeMessagingHosts\com.cwi.engine` pointing to `C:\path\to\com.cwi.engine.json` (with `"path"` set to `launcher.bat`).

---

## 5. Font Subset & FFmpeg CLI Tooling Commands

1. **Subsetting Roboto Flex Variable Font (`307 KB` full $\to$ `62 KB` `wdth,wght` subset):**
   If rebuilding font assets from source using `fonttools` ([Stack Overflow](https://stackoverflow.com/questions/76091800/subset-a-variable-font-to-only-include-required-axis)):
   ```bash
   pip install fonttools brotli
   # Restrict variable axes to wdth and wght while dropping unused parametric axes for ultra-light 62 KB bundle:
   python3 -m fontTools.varLib.instancer RobotoFlex-Full.ttf \
     GRAD=0 XOPQ=96 XTRA=468 YOPQ=79 YTAS=750 YTDE=-203 YTFI=738 YTLC=514 YTUC=712 opsz=14 slnt=0 \
     -o RobotoFlex-wdth-wght.ttf
   fonttools subset RobotoFlex-wdth-wght.ttf \
     --unicodes="U+0020-007E,U+00A0-00FF,U+2000-206F" \
     --flavor=woff2 \
     --output-file=frontend_renderer/assets/fonts/RobotoFlex-Subset-wdth-wght.woff2
   ```
2. **FFmpeg `libass` Offline Burn-In Command (`frontend_renderer/offline_burnin/ffmpeg_burnin.py`):**
   ```bash
   ffmpeg -y -i input_video.mp4 \
     -vf "ass=output_captions.ass:fontsdir=frontend_renderer/assets/fonts" \
     -c:v libx264 -preset fast -crf 18 -c:a copy \
     output_cwi_burned.mp4
   ```
