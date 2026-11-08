from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
try:
    import psutil
except ImportError:
    psutil = None
import torch


ProfileTier = Literal[
    "TIER_1_CUDA_HIGH",
    "TIER_1B_CUDA_6GB",
    "TIER_2_APPLE_SILICON",
    "TIER_4_CPU_VULKAN",
]

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_media_path(filename: str, default_win_dir: str = "D:/dune") -> Path:
    """
    Resolves media and CaptionSpec files across environment variables, repo `data/`,
    `frontend_renderer/public/`, `backend/ml_engine/`, and `D:/dune/` so the platform
    runs on Windows, macOS, Linux, and CI pipelines without hardcoded path crashes.
    """
    env_dir = os.environ.get("CWI_DATA_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir) / filename)
    candidates.extend(
        [
            REPO_ROOT / "data" / filename,
            REPO_ROOT / "frontend_renderer" / "public" / filename,
            REPO_ROOT / "backend" / "ml_engine" / filename,
            Path(default_win_dir) / filename,
            Path(filename),
        ]
    )
    for c in candidates:
        if c.exists():
            return c
    return REPO_ROOT / "data" / filename


@dataclass
class EngineConfig:
    tier: ProfileTier
    device: str
    compute_type: str
    asr_model_id: str
    asr_draft_model_id: Optional[str]
    use_speculative_decoding: bool
    diarizer_backend: Literal["sortformer_v2", "pyannote_streaming", "lightweight_512mb"]
    sortformer_profile: Literal["low_latency_1_04s", "Offline_30_4s"]
    whistress_checkpoint: str
    sensevoice_model_id: str
    emotion2vec_model_id: str
    huggingface_token: Optional[str] = os.environ.get("HUGGINGFACE_TOKEN")
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
    if psutil is not None:
        sys_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    else:
        sys_ram_gb = 16.0
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
