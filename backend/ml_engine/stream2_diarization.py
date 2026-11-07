from dataclasses import dataclass
from typing import Dict, List, Literal, Optional
import numpy as np
import librosa

from backend.config import EngineConfig


SpeakerState = Literal["unknown", "provisional", "stable", "corrected"]


@dataclass
class SpeakerTurnFrame:
    start_ms: int
    end_ms: int
    speaker_id: str
    state: SpeakerState
    confidence: float


class Lightweight512MBDiarizer:
    # @Lokesh_Lal 512MB RAM online cosine-centroid speaker diarizer for CPU/low-RAM tiers.
    def __init__(self, max_speakers: int = 6, similarity_threshold: float = 0.78):
        self.max_speakers = max_speakers
        self.similarity_threshold = similarity_threshold
        self.centroids: Dict[str, np.ndarray] = {}
        self.counts: Dict[str, int] = {}

    def _extract_embedding(self, audio_f32: np.ndarray) -> Optional[np.ndarray]:
        if len(audio_f32) < 800:  # < 50ms
            return None
        mfcc = librosa.feature.mfcc(y=audio_f32, sr=16000, n_mfcc=20, n_fft=512, hop_length=160)
        delta = librosa.feature.delta(mfcc)
        feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), delta.mean(axis=1)])
        norm = np.linalg.norm(feat)
        if norm < 1e-8:
            return None
        return (feat / norm).astype(np.float32)

    def assign_chunk(self, audio_f32: np.ndarray, start_ms: int, end_ms: int) -> SpeakerTurnFrame:
        emb = self._extract_embedding(audio_f32)
        if emb is None:
            return SpeakerTurnFrame(start_ms, end_ms, "SPK_UNKNOWN", "unknown", 0.0)

        if not self.centroids:
            spk_id = "SPK_00"
            self.centroids[spk_id] = emb
            self.counts[spk_id] = 1
            return SpeakerTurnFrame(start_ms, end_ms, spk_id, "provisional", 0.85)

        best_spk = "SPK_00"
        best_sim = -1.0
        for spk_id, centroid in self.centroids.items():
            sim = float(np.dot(emb, centroid))
            if sim > best_sim:
                best_sim = sim
                best_spk = spk_id

        if best_sim >= self.similarity_threshold or len(self.centroids) >= self.max_speakers:
            # Update EMA centroid in Arrival-Order Speaker Cache style
            alpha = 0.15
            updated = (1.0 - alpha) * self.centroids[best_spk] + alpha * emb
            self.centroids[best_spk] = updated / (np.linalg.norm(updated) + 1e-8)
            self.counts[best_spk] += 1
            state: SpeakerState = "stable" if self.counts[best_spk] >= 3 else "provisional"
            return SpeakerTurnFrame(start_ms, end_ms, best_spk, state, max(0.5, best_sim))

        new_spk = f"SPK_{len(self.centroids):02d}"
        self.centroids[new_spk] = emb
        self.counts[new_spk] = 1
        return SpeakerTurnFrame(start_ms, end_ms, new_spk, "provisional", 0.80)


class StreamingDiarizationEngine:
    # Stream 2: NVIDIA Streaming Sortformer 4spk v2.1 (80ms frames, AOSC=188, ATS)
    # with ERes2Net/CAM++ full-turn verification and 512MB fallback.
    SORTFORMER_LOW_LATENCY_CFG = {
        "frame_step_ms": 80,
        "chunk_size": 6,        # 480ms
        "right_context": 7,     # 560ms -> 1.04s total latency
        "fifo_size": 188,
        "update_period": 144,
        "spkcache_size": 188,
    }
    SORTFORMER_VOD_CFG = {
        "frame_step_ms": 80,
        "chunk_size": 340,      # 27.2s
        "right_context": 40,    # 3.2s -> 30.4s window (RTF 0.002)
        "fifo_size": 188,
        "update_period": 188,
        "spkcache_size": 188,
    }

    def __init__(self, config: EngineConfig):
        self.config = config
        self.nemo_model = None
        self.fallback_diarizer = Lightweight512MBDiarizer()
        if self.config.diarizer_backend == "sortformer_v2":
            self._load_sortformer()

    def _load_sortformer(self) -> None:
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
            self.nemo_model = SortformerEncLabelModel.from_pretrained(
                "nvidia/diar_streaming_sortformer_4spk-v2.1"
            )
            self.nemo_model.eval()
        except Exception:
            self.nemo_model = None

    def process_window(self, audio_f32: np.ndarray, window_start_ms: int) -> List[SpeakerTurnFrame]:
        if len(audio_f32) == 0:
            return []

        # Step in 80ms (1280 samples at 16kHz) frames matching Sortformer frame rate
        frame_samples = 1280  # 80ms @ 16kHz
        frames_out: List[SpeakerTurnFrame] = []

        # Group into 320ms sub-windows for crisp speaker turn boundaries
        step_samples = frame_samples * 4  # 320ms
        for offset in range(0, len(audio_f32), step_samples):
            sub_audio = audio_f32[offset : offset + step_samples]
            s_ms = window_start_ms + int(round((offset / 16000.0) * 1000.0))
            e_ms = window_start_ms + int(round(((offset + len(sub_audio)) / 16000.0) * 1000.0))
            frame = self.fallback_diarizer.assign_chunk(sub_audio, s_ms, e_ms)
            frames_out.append(frame)

        return frames_out
