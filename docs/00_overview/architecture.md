# System Architecture & End-to-End Engineering Topology (`architecture.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [Weave (SNU, 2026)](https://xlaude2040.com/weave/), [CwiRenderer](https://stardance.hackclub.com/projects/63123), [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper), [NVIDIA Streaming Sortformer v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)

---

## 1. High-Level Architectural Overview

To support both **real-time streaming inference** (via a Chrome Manifest V3 extension and optional wearable microphone array) and **frame-accurate offline VOD rendering** (via an HTML5/React 19 desktop video player and FFmpeg `.ass` burn-in pipeline), the platform is architected as **two decoupled subsystems** connected by a strictly versioned JSON data contract (`CaptionSpec`, `schema_version: "2.0-cwi"`) ([Weave](https://xlaude2040.com/weave/), [CwiRenderer](https://stardance.hackclub.com/projects/63123)):

1. **Subsystem 1 — The Parallel ML Audio Engine (`backend/ml_engine/` & `extension/workers/webgpu_worker.js`):**
   - Ingests raw `16,000 Hz`, mono, 32-bit float (or 16-bit signed integer) PCM audio.
   - Applies Stage 0 pre-filtering via **[Silero VAD](https://github.com/MohammdKopa/kami-subs)** (`10–20 ms` frames, `SILENCE_RMS` gate) and optional **[SCNet](https://huggingface.co/papers?q=CPU+inference)** / **[HT-Demucs v4](https://ai.miraheze.org/wiki/UVR5)** vocal stem isolation.
   - Executes **four concurrent extraction streams** joined deterministically by word boundary timestamps $[t_{\text{start}}, t_{\text{end}}]$:
     - **Stream 1 (Verbatim ASR & Sub-50ms Alignment):** [CrisperWhisper 2.0](https://github.com/nyrahealth/CrisperWhisper) (`29.6 ms` TIMIT mean absolute boundary error via Viterbi cross-attention alignment + `turbo` speculative decoding) or live [Nemotron ASR + Parakeet TDT v3](https://xlaude2040.com/weave/) (`0.62s` median streaming word emission) or in-browser [Moonshine Base](https://webai.show/models/moonshine-asr/).
     - **Stream 2 (Streaming Speaker Diarization):** [NVIDIA Streaming Sortformer 4spk v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) (`80 ms` frames, `1.04s` low-latency profile, Arrival-Time Ordering + Arrival-Order Speaker Cache) + full-turn ERes2Net/CAM++ identity verification, or [Pyannote Streaming](https://www.pyannote.ai/blog/how-we-built-streaming-diarization) (`100 ms` chunks), or [512 MB RAM custom clustering](https://stardance.hackclub.com/projects/63123).
     - **Stream 3 (Acoustic Prosody & Layer-9 Sentence Stress):** Frame-level $F_0$ pitch and RMS loudness (`75 ms` window, `20 ms` stride, `300 ms` pooling, speaker running median $F_0$ tracker) + **[WhiStress](https://www.isca-archive.org/interspeech_2025/yosha25_interspeech.pdf)** (`7M`-parameter cross-attention + FCNN head tapped at Layer 9 of the frozen Whisper backbone).
     - **Stream 4 (Paralinguistic Events, SFX & Vocal Texture):** [SenseVoice-Small / Large](https://fun-audio-llm.github.io/) (`~15x` faster than Whisper-large-v3) for `<|BGM|>`, `<|Applause|>`, `[laughter]`, `[crying]`, `[sighs]`, `[gasps]` + **[emotion2vec+](https://huggingface.co/emotion2vec/emotion2vec_plus_large)** (`50 Hz` frame-level affect embeddings) or **[emotion2vec-web-distill](https://huggingface.co/thomashallock/emotion2vec-web-distill)** (`2.23M` params for WebGPU).

2. **Subsystem 2 — The Deterministic Rendering & Visualization Engine (`frontend_renderer/` & `extension/content/`):**
   - Resolves the causality mismatch between acoustic onset ($t_0$) and ML inference latency by maintaining a **Dual-Clock Read-Ahead Buffer** (`presentation_offset_ms = 1750`, minimum floor `420 ms`) ([Weave](https://xlaude2040.com/weave/)).
   - Mounts an isolated **Shadow DOM** overlay (`overlay.js`) inside the lower 20% screen work area with a 90% opacity black backdrop ([CwiRenderer](https://stardance.hackclub.com/projects/63123)).
   - Binds each word's pre-computed [Roboto Flex](https://www.csstypestudio.com/library/font/roboto-flex) variable font axes (`wght`, `wdth`, `GRAD`, `opsz`, `slnt`, `XTRA`, `XOPQ`, `YOPQ`, `YTAS`, `YTDE`) and kinetic trajectories once upon entering the read-ahead zone, using a **Single-Delay CSS Scheduling Pattern** (`animation-delay: calc(var(--t-start-ms) - var(--t-present-ms))`) so the browser's hardware compositor triggers the letter-by-letter **Colour Turn**, **1.15× Pop**, **Crest** size envelope, and **-0.525em Hold** lift at 60–120 fps with zero JavaScript timer jitter ([Weave](https://xlaude2040.com/weave/)).
   - Provides a deterministic offline **`.ass` Sync Engine & FFmpeg `libass` Burn-In Pipeline** verified against 256 unit test invariants ([CwiRenderer](https://stardance.hackclub.com/projects/63123)).

---

## 2. End-to-End Dataflow Diagram (ASCII Blueprint for Code Generators)

```text
+---------------------------------------------------------------------------------------------------+
|                                   AUDIO INGESTION & CAPTURE                                       |
|  [Chrome Tab (tabCapture)] ---> [offscreen.js: Mandatory AudioContext Speaker Passthrough]        |
|                                         |                                                         |
|  [Local Media / Mic Array] ------------>+---> [pcm-worklet.js: Stereo->Mono, 16kHz PCM32/PCM16]   |
+---------------------------------------------------------------------------------------------------+
                                                  |
                  +-------------------------------+-------------------------------+
                  | (Mode A: WebSocket ws://127.0.0.1:8765/ws)                    | (Mode B: Zero-Copy MessageChannel)
                  v                                                               v
+---------------------------------------------------------------+   +-------------------------------+
| SUBSYSTEM 1A: PYTHON FASTAPI ML ENGINE (backend/ml_engine/)   |   | SUBSYSTEM 1B: WEBGPU WORKER   |
|                                                               |   | (extension/workers/)          |
|  [Stage 0: Audio Pre-Processor]                               |   |  - Silero VAD (ONNX)          |
|   |-- Silero VAD (10-20ms window, SILENCE_RMS gate)           |   |  - Moonshine Base (65MB q8)   |
|   +-- Optional SCNet / HT-Demucs v4 (Vocals vs BGM/SFX stems) |   |    or Whisper Base (145MB)    |
|          |                                                    |   |  - 75ms/20ms F0 & RMS DSP     |
|          +---> [Stream 1: Verbatim ASR & Sub-50ms Aligner]    |   |  - emotion2vec-web-distill    |
|          |      - VOD/Buffered: CrisperWhisper 2.0 (ct2 +     |   |    (2.23M params, 50Hz)       |
|          |        turbo speculative, 29.6ms Viterbi align)    |   +-------------------------------+
|          |      - Live: Nemotron ASR (0.62s) + Parakeet TDT   |                 |
|          |                                                    |                 |
|          +---> [Stream 2: Streaming Speaker Diarization]      |                 |
|          |      - Streaming Sortformer 4spk v2.1 (80ms frames,|                 |
|          |        ATS + AOSC, 1.04s latency) + ERes2Net/CAM++ |                 |
|          |      - Fallback: Pyannote (100ms) or 512MB Diarizer|                 |
|          |                                                    |                 |
|          +---> [Stream 3: Prosody & Layer-9 Stress Head]      |                 |
|          |      - 75ms/20ms F0 & RMS -> 300ms pool            |                 |
|          |      - Speaker Running Median F0 Tracker           |                 |
|          |      - WhiStress 7M Head on Whisper Layer 9        |                 |
|          |                                                    |                 |
|          +---> [Stream 4: Paralinguistics & Vocal Texture]    |                 |
|                 - SenseVoice-Small/Large (<|BGM|>, [sighs])   |                 |
|                 - emotion2vec+ (50Hz frame arousal/texture)   |                 |
|                                                               |                 |
|  [Timestamp Joiner & Typography Synthesizer]                  |                 |
|   +-- Merges Streams 1-4 on [t_start, t_end]                  |                 |
|   +-- Computes wght (340-900), wdth (25-151), Crest (0.72-1.62)|                |
+---------------------------------------------------------------+                 |
                                  |                                               |
                                  +-----------------------+-----------------------+
                                                          |
                                                          v
                                     +-----------------------------------------+
                                     |   CaptionSpec v2.0-cwi JSON Contract    |
                                     +-----------------------------------------+
                                                          |
                  +---------------------------------------+---------------------------------------+
                  | (Live / Interactive DOM Stream)                                               | (Offline Export)
                  v                                                                               v
+-----------------------------------------------------------------------+   +-----------------------------------------+
| SUBSYSTEM 2A: CwiRenderer DOM/CSS ENGINE (frontend_renderer/ & ext/)  |   | SUBSYSTEM 2B: OFFLINE ASS & FFMPEG      |
|  - Dual-Clock Read-Ahead Buffer (t_present = t_acoustic - 1750ms)     |   |  - Sync Engine -> .ass Karaoke (\k)     |
|  - Lower 20% Work Area + 90% Black Box (rgba(0,0,0,0.90))             |   |  - \t(\fscx\fscy) Pop & Crest Envelopes |
|  - Immutable Post-Onset Text Lock (freezes DOM once t_present >= t0)  |   |  - FFmpeg libass Hardware/CPU Burn-In   |
|  - Single-Delay CSS Keyframes: Colour Turn, 1.15x Pop, Crest (0.70    |   |  - Verified by 256 Unit Test Invariants |
|    sustain), Weight (340-900), Hold (-0.525em), Voice Compass Ring    |   +-----------------------------------------+
+-----------------------------------------------------------------------+
```

---

## 3. The Dual-Clock Read-Ahead Causality Resolution (`-1.75s` Offset)

A fundamental physical constraint of expressive captioning is **causality mismatch** ([Weave](https://xlaude2040.com/weave/)):
- Streaming ASR emits provisional words at a median latency of **0.62 seconds** after vocal onset ($t_0$).
- Furthermore, a word's complete pitch contour ($F_0$), peak RMS loudness, and spoken duration ($\Delta t = t_{\text{end}} - t_{\text{start}}$) cannot be physically measured until the speaker finishes pronouncing the word.
- If captions were rendered at zero acoustic delay ($t_{\text{present}} = t_{\text{acoustic}}$), the word-onset color wipe and `1.15×` Pop would trigger `620+ ms` *after* the actor's lips moved, destroying synchronization.

### How the Dual-Clock Architecture Solves Causality:
1. **Acoustic Clock ($t_{\text{acoustic}}$):** Tracks the raw incoming audio capture stream at real-time $t$.
2. **Presentation Clock ($t_{\text{present}}$):** Runs **1,750 ms behind** the acoustic clock:
   $$t_{\text{present}} = t_{\text{acoustic}} - 1750\text{ ms}$$
   *(For buffered VOD or local video files, the engine simply reads the audio buffer `1.75s` ahead of the video `<video>.currentTime` without delaying video playback; for live calls/streams, either a `1.75s` media playback buffer delay is applied or the read-ahead window dynamically clamps down to the **420 ms minimum floor**).*
3. **Timeline of a Single Word Token ($t_0 = 10,000\text{ ms}$ acoustic):**
   - **$t_{\text{acoustic}} = 10,000\text{ ms}$ ($t_{\text{present}} = 8,250\text{ ms}$):** Actor begins speaking the word.
   - **$t_{\text{acoustic}} = 10,620\text{ ms}$ ($t_{\text{present}} = 8,870\text{ ms}$):** Streaming ASR ([Nemotron](https://xlaude2040.com/weave/) or [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper)) emits the word token along with its `300 ms` pooled $F_0$, RMS, and Layer-9 prominence. The word immediately renders inside the lower 20% caption box in **uncolored neutral read-ahead grey** (`#B0B5BC`).
   - **$t_{\text{acoustic}} \in [10,620\text{ ms}, 11,750\text{ ms})$ ($t_{\text{present}} \in [8,870\text{ ms}, 10,000\text{ ms})$):** Viewer reads the word comfortably in the read-ahead zone for **1.13 seconds** (`1130 ms > 420 ms` minimum floor). Any late endpoint text verification from [Parakeet TDT v3](https://xlaude2040.com/weave/) or speaker attribution stabilization from [Streaming Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) updates the word silently while it is still grey.
   - **$t_{\text{acoustic}} = 11,750\text{ ms}$ ($t_{\text{present}} = 10,000\text{ ms} = t_0$):** The presentation playhead crosses $t_0$.
     - **Text Freeze:** The word's text and DOM box geometry lock permanently (**Immutable Text Rule**).
     - **Kinetic Trigger:** The pre-scheduled CSS animation fires at exact millisecond alignment with the video frame: the `1.15×` Pop fires, `Crest` scales to peak and settles to `0.70` peak, `wght` and `wdth` lock to the speaker's vocal delivery, and the **Colour Turn** wipes left-to-right across the characters over $\Delta t = t_{\text{end}} - t_{\text{start}}$.

---

## 4. Four Local Hardware Execution Profiles

The architecture runs **100% locally offline** across four hardware tiers plus an optional wearable edge companion ([CyberPulsTech](https://cyberpulstech.com/local-whisper-ai-speech-to-text-gpu-vram-benchmarks/), [GigaGPU](https://gigagpu.com/whisper-rtf-by-gpu/), [Weave](https://xlaude2040.com/weave/)):

1. **Profile 1 — Consumer NVIDIA GPU (`cuda_ct2`):**
   - **Target Hardware:** NVIDIA RTX 3060 12GB, RTX 4060 8GB, RTX 4070 12GB, RTX 3090/4090/5080/5090.
   - **Runtime:** Python FastAPI + CTranslate2 (`ct2`) speculative decoding (`CrisperWhisper 2.0 large` + `turbo` draft, or `Faster-Whisper large-v3-turbo`) + `Streaming Sortformer v2.1` + `WhiStress Layer-9` + `SenseVoice-Small` + `emotion2vec+`.
   - **Performance:** `~2.9–5.8 GB` VRAM, `1.5x` to `40.9x` real-time speed (`RTF 0.002` to `0.10`), `<250 ms` first-segment latency.
2. **Profile 2 — Apple Silicon Unified Memory (`apple_mlx_coreml`):**
   - **Target Hardware:** MacBook Air/Pro, Mac Mini/Studio (M1 / M2 / M3 / M4 with `8 GB–36 GB` Unified RAM).
   - **Runtime:** `whisper.cpp` (with Apple Neural Engine Core ML encoder acceleration `>3x` faster than CPU, plus Metal GPU decoder) or `Nemotron + Parakeet` + Swift Core ML `Streaming Sortformer` + `mlx-community/emotion2vec-plus-large-mlx` (`fp16`).
   - **Performance:** `~2.1–4.8 GB` Unified RAM, `2.5x` to `50.0x` real-time speed (`72s` per hour of audio on M3 Max).
3. **Profile 3 — Zero-Server In-Browser WebGPU (`browser_webgpu`):**
   - **Target Hardware:** Any desktop/laptop running Chrome/Edge 113+ or Safari 26+.
   - **Runtime:** 100% inside Chrome Extension Web Workers via `@huggingface/transformers` (ONNX Runtime Web) running `onnx-community/moonshine-base-ONNX` (`~65 MB` `q8`) or `Whisper Base` (`~145–210 MB` `q4f16`/`q8`) + `Silero VAD` + `thomashallock/emotion2vec-web-distill` (`2.23M` params).
   - **Performance:** `5x–15x` real-time speed within Chrome's `1–4 GB` WebGPU tab memory ceiling and `128 MB` buffer binding limit.
4. **Profile 4 — Low-Resource x86 CPU / Vulkan iGPU + Optional Wearable Pi Node (`cpu_int8_vulkan`):**
   - **Target Hardware:** Intel Core i7 / AMD Ryzen 7 CPU-only laptops or Radeon 680M / Intel Arc iGPUs.
   - **Runtime:** `Parakeet TDT v3` (`~1.5 GB` RAM, `30x` RT on i7-12700KF) or `Faster-Whisper small INT8` (`~852 MB–2.2 GB` RAM, `7.8x` RT) or `whisper.cpp 1.8.3 Vulkan` (`~12x` iGPU boost) + `@Lokesh_Lal`'s custom `512 MB` RAM speaker clustering algorithm.
   - **Optional Wearable Edge Node:** Raspberry Pi Zero 2 W (`512 MB` RAM) + ReSpeaker XVF3800 4-mic array polling Direction-of-Arrival (DoA) every `120 ms` and driving left/right coin vibration motors (`100 ms` pulse per `300 ms` cycle).

---

## 5. Four Parallel Engineering Workstreams (Multi-Agent Decomposition)

To enable conflict-free parallel code generation by smaller models, the repository is partitioned into four independent workstreams synchronized solely by `CaptionSpec v2.0-cwi`:

| Agent ID | Workstream Name | Owned Root Directory | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Agent 1** | **Parallel Audio ML & Prosody Backend** | `backend/` & `wearable_pi_node/` | FastAPI WebSocket/SSE server (`ws://127.0.0.1:8765/ws`), Stage 0 VAD/SCNet, Streams 1–4 orchestrator, F0/RMS DSP calculator, speaker median pitch tracker, `CaptionSpec` builder, and Pi Zero 2 W TCP receiver. |
| **Agent 2** | **Custom Model Training & ONNX Distillation** | `training/` | PyTorch training pipeline for the `7M`-parameter WhiStress Layer-9 head on frozen CrisperWhisper 2.0 (`TinyStress-15K` + `BU-RNC`, target $\ge 0.90$ F1), downstream `emotion2vec+` continuous arousal + 7-class vocalization head (`MSP-Podcast v2.0`, `VocalSound`, `Nonspeech7k`), and hybrid ONNX (`q4f16`/`q8`) exporter. |
| **Agent 3** | **Chrome Manifest V3 Extension & WebGPU Worker** | `extension/` | `manifest.json`, `background.js` (single-session state + Native Messaging `launcher.py`), `offscreen.js` (`chrome.tabCapture` + mandatory `AudioContext` speaker passthrough + WS client), `pcm-worklet.js` (`16 kHz` mono resampler), and serverless `webgpu_worker.js` (`Moonshine Base` + `Silero VAD` + `emotion2vec-web-distill`). |
| **Agent 4** | **Deterministic CwiRenderer, Shadow DOM & FFmpeg Burn-In** | `frontend_renderer/` & `tests/` | Shadow DOM `overlay.js` + standalone React 19 player, bundled `RobotoFlex.woff2` (`307 KB` full / `62 KB` subset) & `NotoSansKR.woff2`, `-1.75s` Dual-Clock scheduler, Single-Delay CSS animations for all 5 kinetic channels, Voice Compass SVG ring, `.ass` Sync Engine + FFmpeg `libass` burn-in CLI, and the 256-invariant test suite. |
