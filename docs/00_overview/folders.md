# Complete Monorepo File Tree & Module Signatures (`folders.md`)

> **Purpose:** This document defines the **exact file tree, module responsibilities, imports, and class/function signatures** for the entire automated "Caption with Intention" repository (`cwi-platform/`). A Flash model must create files at these exact paths with these exact names so all imports across Agents 1, 2, 3, and 4 resolve without error.

---

## 1. Master Monorepo Directory Tree

```text
cwi-platform/
├── pyproject.toml                                  # Python 3.10+ backend, ML & training dependencies
├── requirements.txt                                # Pinned pip requirements for backend & training
├── package.json                                    # Node.js workspace config (extension + frontend_renderer)
├── README.md                                       # Master project setup & execution guide
│
├── shared_schemas/                                 # SHARED CONTRACT ACROSS ALL 4 AGENTS
│   ├── caption_spec.schema.json                    # JSON Schema Draft 2020-12 for CaptionSpec v2.0-cwi
│   ├── speaker_palette.json                        # Official high-contrast CWI speaker color wheel hex codes
│   └── roboto_flex_axes.json                       # Min/Max/Default bounds for all 13 Roboto Flex axes
│
├── backend/                                        # AGENT 1: PYTHON FASTAPI ML & PROSODY ENGINE
│   ├── __init__.py
│   ├── config.py                                   # Hardware profile auto-detection & runtime constants
│   ├── main.py                                     # FastAPI app, WebSocket (ws://127.0.0.1:8765/ws) & SSE endpoints
│   ├── native_host/
│   │   ├── com.cwi.engine.json                     # Chrome Native Messaging Host manifest (Linux/macOS/Windows)
│   │   ├── launcher.py                             # Native Messaging stdio bridge that spawns uvicorn backend
│   │   └── launcher.bat                            # Windows wrapper for Chrome Native Messaging
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── caption_spec.py                         # Pydantic v2 models for CaptionSpec v2.0-cwi & WS frames
│   ├── ml_engine/
│   │   ├── __init__.py
│   │   ├── orchestrator.py                         # ParallelAudioPipeline: runs Stage 0 + Streams 1-4 concurrently
│   │   ├── joiner.py                               # TimestampJoiner: aligns Streams 1-4 onto word spans [t0, t1]
│   │   ├── typography_math.py                      # Deterministic F0->wght, RMS->Crest, Hold & GRAD formulas
│   │   ├── stage0_vad_separator.py                 # 16kHz PCM normalizer, Silero VAD gate & SCNet/Demucs separator
│   │   ├── stream1_asr.py                          # CrisperWhisper 2.0 (ct2 speculative) & Nemotron+Parakeet live
│   │   ├── stream2_diarization.py                  # Streaming Sortformer v2.1 (AOSC/ATS), ERes2Net & 512MB fallback
│   │   ├── stream3_prosody_stress.py               # 75ms/20ms F0 & RMS DSP, 300ms pool, Median F0 & WhiStress L9
│   │   └── stream4_paralinguistics.py              # SenseVoice-Small/Large event detector & emotion2vec+ 50Hz affect
│   └── wearable_bridge/
│       ├── __init__.py
│       └── tcp_compass_server.py                   # Async TCP server receiving 120ms DoA bearings from Pi Zero 2 W
│
├── training/                                       # AGENT 2: CUSTOM TRAINING & ONNX DISTILLATION
│   ├── __init__.py
│   ├── configs/
│   │   ├── whistress_layer9_config.yaml            # Hyperparameters for 7M WhiStress head on CrisperWhisper 2.0
│   │   └── emotion2vec_head_config.yaml            # Hyperparameters for continuous arousal & 7-class vocalizations
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── download_and_prepare.py                 # Automated dataset downloader & 16kHz audio normalizer
│   │   ├── tinystress_burnc_dataset.py             # PyTorch Dataset for slprl/TinyStress-15K & BU-RNC ToBI stress
│   │   └── affect_vocalization_dataset.py          # PyTorch Dataset for MSP-Podcast v2.0, MELD, CREMA-D, VocalSound, Nonspeech7k
│   ├── models/
│   │   ├── __init__.py
│   │   ├── whistress_layer9_head.py                # 7M-param Cross-Attention Decoder Block + 2-layer FCNN Classifier
│   │   └── emotion2vec_multitask_head.py           # Attentive pooling + continuous V/A/D & 7-class non-speech head
│   ├── train_whistress_head.py                     # Stage 1 (TinyStress-15K) + Stage 2 (BU-RNC/Buckeye) trainer
│   ├── extract_emotion2vec_features.py             # Dumps 50Hz .npy embeddings from frozen emotion2vec_plus_large
│   ├── train_emotion2vec_head.py                   # Trains linear/attentive affect & vocalization head
│   └── export_onnx_hybrid.py                       # Exports q4f16/q8 hybrid quantized ONNX models for WebGPU
│
├── extension/                                      # AGENT 3: CHROME MANIFEST V3 EXTENSION & WEBGPU WORKER
│   ├── manifest.json                               # Chrome MV3 manifest (tabCapture, offscreen, nativeMessaging)
│   ├── background.js                               # Service worker: session state, Native Host launcher, offscreen init
│   ├── offscreen/
│   │   ├── offscreen.html                          # Minimal DOM host for AudioContext & WebGPU Worker
│   │   ├── offscreen.js                            # tabCapture stream -> Mandatory Speaker Passthrough + AudioWorklet
│   │   └── pcm-worklet.js                          # AudioWorkletProcessor: stereo->mono downmix & 16kHz resampler
│   ├── workers/
│   │   └── webgpu_worker.js                        # Serverless Transformers.js worker (Moonshine Base + VAD + emotion2vec-web)
│   ├── content/
│   │   ├── content.js                              # Finds <video>, tracks playhead, mounts Shadow DOM CwiRenderer
│   │   └── overlay.css                             # Scoped Shadow DOM styles, @font-face Roboto Flex & Single-Delay keyframes
│   └── popup/
│       ├── popup.html                              # Extension popup UI (Mode switch, Intensity slider, Exception profiles)
│       └── popup.js                                # Persists user settings to chrome.storage.local & notifies content.js
│
├── frontend_renderer/                              # AGENT 4: DETERMINISTIC CwiRenderer, PLAYER & FFMPEG BURN-IN
│   ├── assets/
│   │   ├── fonts/
│   │   │   ├── RobotoFlex-Full.woff2               # 307 KB full 13-axis variable font
│   │   │   ├── RobotoFlex-Subset-wdth-wght.woff2   # 62 KB ultra-fast subset
│   │   │   └── NotoSansKR-Variable.woff2           # Korean Hangul fallback font (weight + scale, no width axis)
│   ├── src/
│   │   ├── core/
│   │   │   ├── dual_clock.ts                       # DualClockController (-1750ms offset, 420ms read-ahead floor)
│   │   │   ├── cwi_renderer.ts                     # Deterministic DOM node reconciler, Immutable Post-Onset Lock
│   │   │   ├── css_scheduler.ts                    # Single-Delay CSS custom property calculator (--cwi-delay, etc.)
│   │   │   ├── voice_compass.ts                    # SVG directional bearing arc renderer for live room mode
│   │   │   └── profiles.ts                         # User customization profiles (Default, HoH, ADHD, Safari wght:404 fix)
│   │   ├── player/
│   │   │   ├── App.tsx                             # Standalone React 19 VOD & Live Captioning Studio
│   │   │   ├── VideoViewport.tsx                   # Video container with lower 20% CWI Shadow DOM overlay
│   │   │   └── ControlPanel.tsx                    # Intensity slider (0-100%), profile selector, .ass export button
│   ├── offline_burnin/
│   │   ├── __init__.py
│   │   ├── sync_engine.py                          # Deterministic frame/timestamp quantizer for offline VOD
│   │   ├── ass_compiler.py                         # Translates CaptionSpec v2.0-cwi to Advanced SubStation Alpha (.ass)
│   │   └── ffmpeg_burnin.py                        # Executes ffmpeg -vf "ass=..." with custom fontdir
│
├── wearable_pi_node/                               # OPTIONAL EDGE COMPANION (RASPBERRY PI ZERO 2 W)
│   ├── pi_neckband_daemon.py                       # Polls ReSpeaker XVF3800 USB mic array every 120ms & drives GPIO motors
│   └── hardware_wiring.md                          # GPIO pinout for dual coin vibration motors & XVF3800 USB interface
│
└── tests/                                          # 256 DETERMINISTIC UNIT & INTEGRATION TESTS
    ├── conftest.py                                 # Pytest fixtures, synthetic audio & sample CaptionSpec payloads
    ├── test_01_schema_contracts.py                 # Tests 1-32: JSON Schema & Pydantic validation invariants
    ├── test_02_typography_math.py                  # Tests 33-80: F0->wght, RMS->Crest, Hold, GRAD & clamp bounds
    ├── test_03_dual_clock_immutability.py          # Tests 81-128: -1.75s read-ahead, 420ms floor & post-onset freeze
    ├── test_04_css_single_delay_channels.py        # Tests 129-176: 5 kinetic channels, 90% black box, lower 20% area
    ├── test_05_ass_sync_and_ffmpeg.py              # Tests 177-224: .ass \k karaoke wipes, \t(\fscx\fscy) & libass burn-in
    └── test_06_ml_streams_and_vad.py               # Tests 225-256: VAD gate, 512MB diarizer, Layer-9 stress & joiner
```

---

## 2. Exact Module-by-Module Signatures & Responsibilities

### 2.1 `backend/` (Agent 1)

| File Path | Key Classes & Functions | Exact Input / Output Signatures |
| :--- | :--- | :--- |
| `backend/config.py` | `HardwareProfile`, `EngineConfig`, `detect_optimal_profile()` | `detect_optimal_profile() -> EngineConfig` (inspects `torch.cuda.is_available()`, `torch.backends.mps.is_available()`, VRAM/RAM size, CPU AVX2/AVX-512 flags). |
| `backend/schemas/caption_spec.py` | `SpeakerInfo`, `ProsodyMetrics`, `TypographySpec`, `WordToken`, `NonSpeechEvent`, `WorkAreaBox`, `CaptionSpec` | Pydantic v2 `BaseModel` classes validating `schema_version == "2.0-cwi"`, enforcing `bottom_pct == 20`, `bg_opacity == 0.90`, `wght` in `[100, 1000]`, `wdth` in `[25, 151]`. |
| `backend/ml_engine/stage0_vad_separator.py` | `AudioPreprocessor` | `process_chunk(pcm_bytes: bytes, sample_rate: int = 16000) -> Tuple[np.ndarray, bool, float, Optional[np.ndarray]]` (returns normalized `float32` mono audio, `is_speech`, `rms_db`, and optional isolated BGM/SFX stem). |
| `backend/ml_engine/stream1_asr.py` | `VerbatimASREngine` | `transcribe_window(audio_f32: np.ndarray, stream_time_ms: int, mode: str = "buffered") -> List[RawWordAlignment]` (runs CrisperWhisper 2.0 `ct2` with `turbo` draft or Nemotron+Parakeet). |
| `backend/ml_engine/stream2_diarization.py` | `StreamingDiarizer`, `LightweightClusterDiarizer` | `process_frame(audio_f32: np.ndarray, timestamp_ms: int) -> List[SpeakerTurnFrame]` (runs 80 ms Sortformer frames + ERes2Net turn verification or 512 MB online cosine-centroid clustering). |
| `backend/ml_engine/stream3_prosody_stress.py` | `ProsodyAndStressExtractor` | `extract_word_prosody(audio_f32: np.ndarray, words: List[RawWordAlignment], encoder_hidden_l9: Optional[torch.Tensor], speaker_id: str) -> List[ProsodyMetrics]` (computes 75ms/20ms $F_0$ & RMS, 300ms pool, updates speaker median $F_0$, runs WhiStress Layer-9 head). |
| `backend/ml_engine/stream4_paralinguistics.py` | `ParalinguisticEventEngine` | `detect_events_and_affect(audio_f32: np.ndarray, bgm_stem: Optional[np.ndarray], start_ms: int) -> Tuple[List[NonSpeechEvent], np.ndarray]` (runs SenseVoice-Small/Large + emotion2vec+ 50 Hz frames). |
| `backend/ml_engine/typography_math.py` | `compute_typography_spec()` | `compute_typography_spec(prosody: ProsodyMetrics, affect_vec: Optional[np.ndarray], intensity: float = 1.0, is_safari: bool = False, locale: str = "en") -> TypographySpec` |
| `backend/ml_engine/joiner.py` | `TimestampJoiner` | `build_caption_spec(segment_id: str, words: List[RawWordAlignment], diar_frames: List[SpeakerTurnFrame], prosody_list: List[ProsodyMetrics], events: List[NonSpeechEvent], compass_data: Optional[Dict]) -> CaptionSpec` |
| `backend/ml_engine/orchestrator.py` | `ParallelAudioPipeline` | `async process_audio_stream(ws: WebSocket) -> AsyncIterator[CaptionSpec]` (runs Streams 1–4 via `asyncio.gather` / threadpool workers). |

### 2.2 `training/` (Agent 2)

| File Path | Key Classes & Functions | Exact Input / Output Signatures |
| :--- | :--- | :--- |
| `training/models/whistress_layer9_head.py` | `WhiStressLayer9Head(nn.Module)` | `forward(decoder_hidden_states: torch.Tensor, encoder_hidden_states_layer9: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor` (outputs `[batch, seq_len, 1]` stress logits; `~7.02M` trainable parameters). |
| `training/models/emotion2vec_multitask_head.py` | `Emotion2VecMultiTaskHead(nn.Module)` | `forward(frame_embeddings_50hz: torch.Tensor) -> Dict[str, torch.Tensor]` (outputs `vad_continuous`: `[B, T, 3]`, `vocalization_logits`: `[B, 7]`, `strain_texture`: `[B, T, 2]`). |
| `training/train_whistress_head.py` | `train_stage1_tinystress()`, `train_stage2_burnc()` | Freezes `nyralabs/CrisperWhisper2.0_large` or `openai/whisper-large-v3-turbo`, hooks Layer 9, masks punctuation tokens (`label = 0`), trains with `BCEWithLogitsLoss`, evaluates F1 on TinyStress-15K (`>= 0.909`) and Aix-MARSEC (`>= 0.961`). |
| `training/export_onnx_hybrid.py` | `export_and_quantize_hybrid()` | Exports encoder in `fp32` and quantizes decoder to `q4f16` (WebGPU) and `q8` (WASM+SIMD) using `onnxruntime.quantization`. |

### 2.3 `extension/` (Agent 3)

| File Path | Key Classes & Functions | Exact Input / Output Signatures |
| :--- | :--- | :--- |
| `extension/background.js` | `SessionManager`, `launchNativeBackend()`, `ensureOffscreenDocument()` | Manages MV3 lifecycle, invokes `chrome.runtime.connectNative("com.cwi.engine")`, creates `offscreen.html`, routes `CaptionSpec` messages to active tab's `content.js`. |
| `extension/offscreen/offscreen.js` | `startTabAudioCapture(streamId, mode)` | Calls `navigator.mediaDevices.getUserMedia({ audio: { mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: streamId } } })`, connects `sourceNode.connect(audioCtx.destination)` (**mandatory speaker passthrough**), and connects `AudioWorkletNode('pcm-16k-worklet')`. |
| `extension/offscreen/pcm-worklet.js` | `PCM16kWorkletProcessor extends AudioWorkletProcessor` | Downmixes $N$ channels to mono, resamples from host rate (`44.1/48 kHz`) to `16,000 Hz` via linear/sinc ring buffer, emits `100 ms` (`1,600` sample) chunks over `MessagePort`. |
| `extension/workers/webgpu_worker.js` | `BrowserEdgeMLPipeline` | Loads `onnx-community/moonshine-base-ONNX` (`q8`), `Silero VAD`, and `thomashallock/emotion2vec-web-distill` on `device: "webgpu"`, computes local DSP $F_0$/RMS, emits `CaptionSpec v2.0-cwi`. |

### 2.4 `frontend_renderer/` (Agent 4)

| File Path | Key Classes & Functions | Exact Input / Output Signatures |
| :--- | :--- | :--- |
| `frontend_renderer/src/core/dual_clock.ts` | `DualClockController` | `getPresentationTimeMs(acousticTimeMs: number): number` (`acousticTimeMs - 1750`), `isInReadAheadZone(wordStartMs: number, presentMs: number): boolean` (`wordStartMs - presentMs <= 1750 && wordStartMs > presentMs`). |
| `frontend_renderer/src/core/cwi_renderer.ts` | `CwiRenderer` | `ingestCaptionSpec(spec: CaptionSpec): void`, `tick(videoCurrentTimeMs: number): void` (enforces Immutable Post-Onset Text Lock: never mutates DOM nodes where `presentMs >= word.start_ms`). |
| `frontend_renderer/src/core/css_scheduler.ts` | `bindWordCSSVariables(el: HTMLElement, word: WordToken, presentMs: number, profile: RenderProfile): void` | Sets `--cwi-onset-delay-ms`, `--cwi-duration-ms`, `--cwi-speaker-color`, `--cwi-pop-scale`, `--cwi-crest-peak`, `--cwi-crest-sustain`, `--cwi-wght`, `--cwi-wdth`, `--cwi-grad`, `--cwi-hold-em` once on first sight. |
| `frontend_renderer/offline_burnin/ass_compiler.py` | `compile_captionspec_to_ass(specs: List[CaptionSpec], video_width: int = 1920, video_height: int = 1080) -> str` | Generates valid `[Script Info]`, `[V4+ Styles]`, and `[Events]` with `\k` centisecond karaoke wipes, `\t(t1,t2,\fscx...\fscy...)` scale envelopes, `\pos`, and `90%` opaque black `\3c&H000000&\4a&H1A&` boxes inside the lower 20% margin. |
