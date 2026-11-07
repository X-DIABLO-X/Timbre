# Six Real-Time Inference Optimizations & Latency Engineering (`optimizations.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint (Section 4)](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [Weave (SNU, 2026)](https://xlaude2040.com/weave/), [CrisperWhisper Speculative Decoding](https://github.com/nyrahealth/CrisperWhisper), [GigaGPU Whisper RTF Benchmarks](https://gigagpu.com/whisper-rtf-by-gpu/), [CyberPulsTech GPU VRAM Benchmarks](https://cyberpulstech.com/local-whisper-ai-speech-to-text-gpu-vram-benchmarks/), [Codersera faster-whisper vs whisper.cpp 2026](https://codersera.com/blog/faster-whisper-vs-whisper-cpp-speech-to-text-2026/), [Moonshine ASR](https://webai.show/models/moonshine-asr/), [Mad Devs WebGPU Hybrid Quantization](https://maddevs.io/writeups/running-ai-models-locally-in-the-browser/)

---

## 1. Real-Time Performance Targets

To achieve responsive interactive captioning, the production pipeline enforces two strict latency SLOs ([GigaGPU](https://gigagpu.com/whisper-rtf-by-gpu/)):
1. **Real-Time Factor ($\text{RTF} = \text{Processing Time} \div \text{Audio Duration}$):** Must remain **$< 0.50$** on all supported hardware tiers, and ideally **$< 0.10$** (processing `10 seconds` of speech in `< 1 second`).
2. **Time to First Byte (TTFB) / First-Segment Latency:** Must remain **$< 300\text{–}500\text{ ms}$** (and `< 250 ms` on consumer RTX GPUs running `small` or `turbo`).

---

## 2. Summary Matrix of the 6 Core Empirical Optimizations

| Optimization # | Engineering Technique | Empirical Speedup / Memory Savings | Subsystem Where Implemented |
| :--- | :--- | :--- | :--- |
| **1. Dual-Clock Read-Ahead Buffer** | Runs visual presentation clock **1.75 seconds behind** the acoustic clock (`420 ms` minimum per-word floor). | Absorbs the `0.62s` median streaming ASR emission delay + `1.04s` Sortformer diarization window with **zero perceived lip-sync lag** ([Weave](https://xlaude2040.com/weave/)). | `frontend_renderer/src/core/dual_clock.ts` & `backend/config.py` |
| **2. CTranslate2 (`ct2`) + Speculative Decoding** | Converts CrisperWhisper 2.0 `large` from PyTorch (`RTF 0.091`) to custom `ct2` (`RTF 0.023`, `3.9x` speedup) + pairs with 4-decoder-layer `turbo` draft model (`draft_model="turbo"`). | **5.3× total speedup** (`RTF 0.018` on NVIDIA L40 FP16) with 100% identical transcript quality ([CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper)). | `backend/ml_engine/stream1_asr.py` |
| **3. INT8 & Hybrid Weight Quantization** | Backend `ct2` `int8` / `int8_float16` quantization; Browser `Transformers.js` keeps encoder at `fp32` and quantizes decoder to `q4f16` (WebGPU) or `q8` (WASM+SIMD). | **~35% lower GPU VRAM** on `large-v2` (from `4,525 MB` FP16 down to **`2,926 MB`** INT8) + **2× token throughput** with FP32-equivalent WER ([CyberPulsTech](https://cyberpulstech.com/local-whisper-ai-speech-to-text-gpu-vram-benchmarks/), [Mad Devs](https://maddevs.io/writeups/running-ai-models-locally-in-the-browser/)). | `backend/config.py`, `training/export_onnx_hybrid.py` & `extension/workers/webgpu_worker.js` |
| **4. Dynamic Input Scaling Without 30s Padding** | Replaces Whisper's fixed `30s` zero-padding in browser with **Moonshine Base** (`~61M`, `65 MB` `q8` ONNX) + Silero VAD sentence-aware buffer flushing (`KAMI_SENT_MAX_CHUNKS=2`, `KAMI_SENT_MAX_CHARS=160`). | Compute scales linearly with actual utterance length; **30%–50% total transcription time reduction** ([Moonshine ASR](https://webai.show/models/moonshine-asr/), [kami-subs](https://github.com/MohammdKopa/kami-subs)). | `extension/workers/webgpu_worker.js` & `backend/ml_engine/stage0_vad_separator.py` |
| **5. Fixed-Chunk Cache Diarization** | Replaces multi-second sliding-window batch re-clustering (legacy DIART) with **80 ms frame updates** into Sortformer's `188`-frame Arrival-Order Speaker Cache (`AOSC`) or stateless **100 ms Pyannote chunks**. | Eliminates redundant sliding-window re-computation; hits `RTF 0.093` live / `RTF 0.002` VOD ([Sortformer v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1), [Pyannote](https://www.pyannote.ai/blog/how-we-built-streaming-diarization)). | `backend/ml_engine/stream2_diarization.py` |
| **6. Apple ANE Core ML & Vulkan iGPU Offloading** | Deploys `whisper.cpp` with **Core ML ANE encoder acceleration** + Metal decoder on macOS, and `whisper.cpp 1.8.3` with **Vulkan** on AMD Radeon 680M / Intel Arc iGPUs. | **>3× faster** than CPU on Apple Silicon (`2x+` on turbo with Metal); **~12× performance boost** on integrated AMD/Intel GPUs via Vulkan ([Codersera](https://codersera.com/blog/faster-whisper-vs-whisper-cpp-speech-to-text-2026/)). | `backend/config.py` & `scripts/setup_native_accelerators.sh` |

---

## 3. Deep-Dive Implementation Details for Each Optimization

### 3.1 Optimization 1: Dual-Clock Read-Ahead Buffer (`-1750 ms` Offset & `420 ms` Floor)
```typescript
// Exact timing invariant enforced in frontend_renderer/src/core/dual_clock.ts:
export const PRESENTATION_OFFSET_MS = 1750;
export const MIN_READ_AHEAD_FLOOR_MS = 420;

export function computeEffectiveOnsetDelayMs(
  wordStartMs: number,
  acousticArrivalMs: number,
  presentationOffsetMs: number = PRESENTATION_OFFSET_MS
): number {
  const presentationClockAtArrival = acousticArrivalMs - presentationOffsetMs;
  const rawReadAheadDuration = wordStartMs - presentationClockAtArrival;
  // Guarantee the viewer always has at least 420ms to read the word in neutral grey before onset wipe
  return Math.max(MIN_READ_AHEAD_FLOOR_MS, rawReadAheadDuration);
}
```

### 3.2 Optimization 2: CTranslate2 (`ct2`) + Speculative Decoding (`5.3×` Speedup)
On an NVIDIA L40 GPU (`fp16`), [CrisperWhisper 2.0](https://github.com/nyrahealth/CrisperWhisper) benchmarks demonstrate:
- HuggingFace PyTorch `nyralabs/CrisperWhisper2.0_large`: **`RTF = 0.091`**
- Custom CTranslate2 (`ct2`) backend standalone: **`RTF = 0.023`** (**`3.9×` speedup**)
- Custom CTranslate2 (`ct2`) with 4-decoder-layer `turbo` speculative draft model (`draft_model="turbo"`): **`RTF = 0.018`** (**`5.3×` total speedup over PyTorch** with mathematically lossless output distribution).
- Coupled with `model.transcribe_dual()`, both verbatim (`[um]`, `[uh]`) and intended transcripts are decoded in a single batched pass at **~1.9× the speed** of two separate passes.

### 3.3 Optimization 4: Sentence-Aware VAD Buffer Flushing (`KAMI_SENT_MAX_CHUNKS=2`, `KAMI_SENT_MAX_CHARS=160`)
Following [kami-subs](https://github.com/MohammdKopa/kami-subs), the backend ring buffer flushes immediately when any of the following three conditions is met:
1. Terminal sentence punctuation (`.`, `?`, `!`) is emitted by the provisional decoder;
2. `chunk_count >= KAMI_SENT_MAX_CHUNKS` (`2` consecutive speech chunks); or
3. `accumulated_chars >= KAMI_SENT_MAX_CHARS` (`160` characters, preventing lower-20% caption box overflow).

### 3.4 Optimization 6: Apple Neural Engine (ANE) Core ML & Vulkan Build Script (`scripts/setup_native_accelerators.sh`)
```bash
#!/usr/bin/env bash
set -euo pipefail

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"

if [[ "$OS_NAME" == "Darwin" && "$ARCH_NAME" == "arm64" ]]; then
  echo "[CWI] Building whisper.cpp with Apple Neural Engine (Core ML) & Metal..."
  git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git /tmp/whisper.cpp
  cd /tmp/whisper.cpp
  cmake -B build -DWHISPER_COREML=1 -DWHISPER_METAL=1
  cmake --build build -j --config Release
elif [[ "$OS_NAME" == "Linux" ]]; then
  echo "[CWI] Building whisper.cpp 1.8.3 with Vulkan iGPU offloading (~12x iGPU boost)..."
  git clone --depth 1 --branch v1.8.3 https://github.com/ggerganov/whisper.cpp.git /tmp/whisper.cpp
  cd /tmp/whisper.cpp
  cmake -B build -DGGML_VULKAN=1
  cmake --build build -j --config Release
fi
```
