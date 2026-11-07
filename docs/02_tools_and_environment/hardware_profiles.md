# Local Hardware Compatibility Matrix & Automatic Runtime Selector (`hardware_profiles.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint (Section 6)](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [GigaGPU Whisper RTF Benchmarks](https://gigagpu.com/whisper-rtf-by-gpu/), [CyberPulsTech GPU VRAM Benchmarks](https://cyberpulstech.com/local-whisper-ai-speech-to-text-gpu-vram-benchmarks/), [SnailText CPU/Mac Benchmarks](https://snailtext.app/blog/do-you-need-a-gpu-for-voice-to-text/), [OfflineTTS Browser WebGPU Benchmarks](https://offlinetts.com/blog/browser-speech-recognition-whisper-comparison/)

---

## 1. Component-by-Component Memory Footprint Reference

1. **ASR Backbone Memory Footprints ([CyberPulsTech](https://cyberpulstech.com/local-whisper-ai-speech-to-text-gpu-vram-benchmarks/), [SnailText](https://snailtext.app/blog/do-you-need-a-gpu-for-voice-to-text/)):**
   - **Whisper Tiny (`39M`):** `~0.5 GB` FP16 VRAM (`75 MB` GGML disk / `~273 MB` RAM, `~7.7%` English WER).
   - **Whisper Base (`74M`):** `~0.8 GB` FP16 VRAM (`142 MB` disk / `~388 MB` RAM, `~5.8%` WER).
   - **Moonshine Base (`~61M`):** `~65 MB` `q8` ONNX disk (`~180 MB` WebGPU RAM).
   - **Whisper Small (`244M`):** `~1.5 GB` FP16 VRAM (`466 MB` disk / `~852 MB` GGML RAM or `2,257 MB` CPU FP32 RAM, `~4.2%` WER).
   - **Whisper Medium (`769M`):** `~3.0 GB` FP16 VRAM (`1.5 GB` disk / `~2.1 GB` RAM, `~3.5%` WER).
   - **Whisper Large-v3-Turbo (`809M`):** `~1.6 GB` CTranslate2 disk, runs in `4 GB+` VRAM (`2.628%` German WER / `<3.5%` English WER).
   - **Whisper Large-v3 / CrisperWhisper 2.0 Large (`1.55B`):** `4.3–5.8 GB` FP16 VRAM (`4,525 MB` FP16 vs. `2,926 MB` INT8 on large-v2; `2.9 GB` GGML disk / `~3.9 GB` RAM, `~2.9%` WER).
   - **Parakeet TDT v3 (`0.6B`):** `~2.3 GB` disk / `~1.5 GB` RAM.
2. **Diarization, Stress & Paralinguistic Footprints:**
   - **WhiStress Layer-9 Head (`7M` params):** Adds `~14 MB` FP16 onto the already-loaded Whisper backbone ([WhiStress](https://www.isca-archive.org/interspeech_2025/yosha25_interspeech.pdf)).
   - **Speaker Diarization:** `Streaming Sortformer v2.1` (`17L NEST + 18L Transformer, hidden_size=192`); `@Lokesh_Lal`'s custom clustering diarizer reduces RAM from `4 GB` down to **`512 MB`** ([CwiRenderer](https://stardance.hackclub.com/projects/63123)).
   - **Affect & Audio Events:** `SenseVoice-Small` (`~15x` faster than Whisper-large-v3), `emotion2vec_plus_base` (`~90M`), `emotion2vec_plus_large` (`~300M`), and `emotion2vec-web-distill` (`2,231,986` parameters, `~9 MB` quantized).

---

## 2. Empirical Hardware Compatibility & Benchmark Matrix

| Tier ID | Device Class & Example Hardware | Selected Runtime & Model Stack | Empirical Speed (RTF) & Latency | Memory Footprint |
| :--- | :--- | :--- | :--- | :--- |
| **`TIER_1_CUDA_HIGH`** | NVIDIA RTX 3060 12GB, RTX 4060 8GB, RTX 4070 12GB, RTX 3090/4090/5080/5090 | **CTranslate2 (`ct2`) Speculative:** `CrisperWhisper 2.0` (`large` + `turbo` draft) or `Faster-Whisper large-v3-turbo` + `WhiStress L9` + `Streaming Sortformer v2.1` + `SenseVoice-Small` + `emotion2vec+` | **RTX 4060:** `RTF 0.10` (small, `210ms`), `RTF 0.30` (medium, `620ms`), `RTF 0.68` (large-v3, `1,450ms`), batch `40.9x RT (88s/hr)`. **RTX 4070 Laptop:** turbo `RTF ~0.002` (`254.7s` in `0.57s`). **RTX 3090:** `125ms` (small) / `850ms` (large-v3, `RTF 0.40`). **RTX 5090:** `350ms` (large-v3, `RTF 0.16`). *(Note: On RTX 3050 6GB, large-v3 is `RTF 1.15`, so auto-select `small` or `turbo` INT8)*. | `2.9 GB` VRAM (INT8) to `4.3–5.8 GB` VRAM (FP16) + `8–32 GB` System RAM |
| **`TIER_2_APPLE_SILICON`** | Apple M1, M2 Pro, M3 Max, M4 (`8 GB–36 GB` Unified RAM) | **Core ML + Metal + MLX:** `whisper.cpp` (ANE Core ML encoder `>3x` faster than CPU) or `Nemotron + Parakeet` + Core ML `Sortformer` + `emotion2vec-plus-large-mlx` (`fp16`) | **M1 CPU:** tiny `25x RT (RTF 0.04)`, base `14x RT (0.07)`, small `6x RT (0.17)`, medium `2.5x RT (0.40)`. **M1 Metal:** large-v3 `1.0x RT`. **M2 Pro Metal:** large-v3 `2.5x RT`. **M3 Max (36GB Metal FP16):** large-v3 `50.0x RT (72s/hr)`. | `~852 MB–2.1 GB` Unified RAM (small/medium) to `3.9–4.8 GB` (large-v3) |
| **`TIER_3_BROWSER_WEBGPU`** | Any Desktop/Laptop with Chrome/Edge 113+ or Safari 26+ | **Transformers.js (ONNX WebGPU):** `Moonshine Base` (`~61M`, `q8` `65 MB`) or `Whisper Base` (`74M`, `~145–210 MB`) + `Silero VAD` + `emotion2vec-web-distill` (`2.23M`) | **WebGPU (`5–10x` faster than WASM):** Whisper Tiny `10–15x RT` (`~120 MB`), Whisper Base `5–8x RT` (`~210 MB`), Whisper Small `2–4x RT` (`~590 MB`). | `<1.5 GB` within Chrome's `1–4 GB` WebGPU tab limit & `128 MB` binding cap |
| **`TIER_4_CPU_VULKAN`** | Intel Core i7-12700K/KF, AMD Ryzen 7/9, or Radeon 680M / Intel Arc iGPU | **CPU AVX2/AVX-512 INT8 or Vulkan iGPU:** `Parakeet TDT v3` or `Faster-Whisper small INT8` or `whisper.cpp 1.8.3 Vulkan` (`~12x` iGPU boost) + `512 MB` custom diarizer | **i7-12700KF:** Parakeet TDT v3 at `30x RT (RTF 0.033)`. **i7-12700K:** Whisper small INT8 at `7.8x RT (RTF 0.13)`. **Ryzen 9 7950X (AVX-512 INT8):** large-v3 at `6.6x RT`. *(Never run large-v3 on older CPUs like Ryzen 7 5700G where `RTF = 3.0`)*. | `1.5 GB` RAM (Parakeet) to `2.2–2.8 GB` RAM + `512 MB` diarizer |
| **`TIER_5_WEARABLE_PI`** | Raspberry Pi Zero 2 W (`512 MB` RAM) + ReSpeaker XVF3800 4-Mic Array | **Edge Sensor & Haptic Node:** USB DoA polling (`120 ms`) + TCP socket to host + Dual Coin Vibration Motors (`100 ms / 300 ms`) | `120 ms` bearing update rate; `<5 ms` LAN TCP latency to host machine. | `<180 MB` of `512 MB` RAM on Pi Zero 2 W |

---

## 3. Production Hardware Auto-Detection Code — `backend/config.py`

Write the following complete implementation to `backend/config.py`:

```python
import os
import platform
from dataclasses import dataclass
from typing import Literal
import psutil
import torch


ProfileTier = Literal[
    "TIER_1_CUDA_HIGH",
    "TIER_1B_CUDA_6GB",
    "TIER_2_APPLE_SILICON",
    "TIER_4_CPU_VULKAN",
]


@dataclass
class EngineConfig:
    tier: ProfileTier
    device: str
    compute_type: str
    asr_model_id: str
    asr_draft_model_id: str | None
    use_speculative_decoding: bool
    diarizer_backend: Literal["sortformer_v2", "pyannote_streaming", "lightweight_512mb"]
    sortformer_profile: Literal["low_latency_1_04s", "Offline_30_4s"]
    whistress_checkpoint: str
    sensevoice_model_id: str
    emotion2vec_model_id: str
    presentation_offset_ms: int = 1750
    read_ahead_floor_ms: int = 420
    vad_silence_rms_threshold: float = 0.008
    kami_sent_max_chunks: int = 2
    kami_sent_max_chars: int = 160
    ws_host: str = "127.0.0.1"
    ws_port: int = 8765
    wearable_tcp_port: int = 8766


def detect_optimal_profile(prefer_offline_vod: bool = False) -> EngineConfig:
    # Deterministically selects the optimal runtime, quantization precision, and model checkpoints
    sys_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    arch = platform.machine().lower()
    sys_name = platform.system().lower()

    # 1. Check NVIDIA CUDA GPU
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        gpu_name = props.name.lower()

        # Guard for 6GB cards (e.g., RTX 3050 6GB where large-v3 hits RTF 1.15)
        if vram_gb < 7.5 or "3050" in gpu_name:
            return EngineConfig(
                tier="TIER_1B_CUDA_6GB",
                device="cuda",
                compute_type="int8_float16",
                asr_model_id="Systran/faster-whisper-small",
                asr_draft_model_id=None,
                use_speculative_decoding=False,
                diarizer_backend="lightweight_512mb",
                sortformer_profile="low_latency_1_04s",
                whistress_checkpoint="weights/whistress_small_layer9.pt",
                sensevoice_model_id="iic/SenseVoiceSmall",
                emotion2vec_model_id="iic/emotion2vec_plus_base",
            )

        # 8GB - 24GB+ Consumer/Pro NVIDIA GPU (RTX 3060 12GB, RTX 4060 8GB, RTX 4070, 3090, 4090, 5090)
        compute_type = "float16" if vram_gb >= 11.5 else "int8_float16"
        return EngineConfig(
            tier="TIER_1_CUDA_HIGH",
            device="cuda",
            compute_type=compute_type,
            asr_model_id="nyrahealth/CrisperWhisper",
            asr_draft_model_id="deepdml/faster-whisper-large-v3-turbo-ct2",
            use_speculative_decoding=True,
            diarizer_backend="sortformer_v2",
            sortformer_profile="Offline_30_4s" if prefer_offline_vod else "low_latency_1_04s",
            whistress_checkpoint="weights/whistress_crisper_layer9.pt",
            sensevoice_model_id="iic/SenseVoiceSmall",
            emotion2vec_model_id="iic/emotion2vec_plus_large",
        )

    # 2. Check Apple Silicon (M1 / M2 / M3 / M4)
    if sys_name == "darwin" and ("arm" in arch or "aarch64" in arch):
        asr_model = "large-v3-turbo" if sys_ram_gb >= 14.0 else "small"
        return EngineConfig(
            tier="TIER_2_APPLE_SILICON",
            device="mps",
            compute_type="float16",
            asr_model_id=f"whisper.cpp-coreml-{asr_model}",
            asr_draft_model_id=None,
            use_speculative_decoding=False,
            diarizer_backend="sortformer_v2" if sys_ram_gb >= 14.0 else "lightweight_512mb",
            sortformer_profile="low_latency_1_04s",
            whistress_checkpoint="weights/whistress_turbo_layer9.pt",
            sensevoice_model_id="iic/SenseVoiceSmall",
            emotion2vec_model_id="mlx-community/emotion2vec-plus-large-mlx",
        )

    # 3. CPU-Only / Integrated GPU Fallback (x86 AVX2/AVX-512 or Vulkan iGPU)
    return EngineConfig(
        tier="TIER_4_CPU_VULKAN",
        device="cpu",
        compute_type="int8",
        asr_model_id="nvidia/parakeet-tdt-0.6b-v2" if sys_ram_gb >= 8.0 else "Systran/faster-whisper-small",
        asr_draft_model_id=None,
        use_speculative_decoding=False,
        diarizer_backend="lightweight_512mb",
        sortformer_profile="low_latency_1_04s",
        whistress_checkpoint="weights/whistress_small_layer9.pt",
        sensevoice_model_id="iic/SenseVoiceSmall",
        emotion2vec_model_id="iic/emotion2vec_plus_base",
    )
```
