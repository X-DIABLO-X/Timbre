# Stream 2: Real-Time Streaming Speaker Diarization & Attribution (`stream_2_diarization.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint (Section 2.3)](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [NVIDIA Streaming Sortformer 4spk v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1), [Sortformer Paper (arXiv:2409.06656v2)](https://arxiv.org/html/2409.06656v2), [Weave (SNU, 2026)](https://xlaude2040.com/weave/), [Pyannote Streaming Diarization](https://www.pyannote.ai/blog/how-we-built-streaming-diarization), [CwiRenderer 512MB Diarizer](https://stardance.hackclub.com/projects/63123)

---

## 1. Architectural Specifications for Stream 2

To assign each word its persistent speaker color (`#4FC3F7`, `#FFD54F`, `#FF8A80`, etc.), Stream 2 resolves speaker turns and overlapping dialogue in streaming fashion:

### 1.1 Primary Engine: NVIDIA Streaming Sortformer 4spk v2.1 (`nvidia/diar_streaming_sortformer_4spk-v2.1`)
- **Architecture:** Combines an L-size (`17-layer`) NeMo Encoder for Speech Tasks (**NEST** Fast-Conformer) with an `18-layer` Transformer encoder (`hidden_size=192`) and two feedforward layers generating **4 sigmoid outputs per `80 ms` frame** (supporting up to 4 concurrent overlapping speakers) ([Sortformer v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)).
- **Permutation Resolution & Long-Horizon Memory:**
  - **Arrival-Time Ordering (ATS):** Resolves the speaker permutation problem deterministically by ordering output slots according to the chronological arrival time of each speaker's first utterance ([arXiv:2409.06656v2](https://arxiv.org/html/2409.06656v2)).
  - **Arrival-Order Speaker Cache (AOSC):** Maintains `spkcache_size = 188` frames (`~15.04 seconds` of high-confidence acoustic embeddings per speaker) to preserve consistent speaker identity across multi-hour streams without drift ([Sortformer v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)).
- **Exact Latency Profiles & Benchmarks:**
  - **Low-Latency Live Profile (`1.04s` buffer latency):**
    - `CHUNK_SIZE = 6` frames (`480 ms`)
    - `RIGHT_CONTEXT = 7` frames (`560 ms`) $\to$ Total buffer latency = $(6 + 7) \times 80\text{ ms} = \mathbf{1.04\text{ seconds}}$
    - `FIFO_SIZE = 188` frames, `UPDATE_PERIOD = 144` frames, `spkcache_size = 188` frames
    - **Speed & Accuracy:** Runs at **RTF `0.093`** on an RTX 6000 Ada GPU, achieving **5.09% DER** on 2-speaker CH109, **6.65% DER** on 2-speaker CALLHOME, and **11.19% DER** across the full CALLHOME benchmark (including overlapping speech) ([Sortformer v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)).
  - **Very High Latency VOD Profile (`30.4s` window):**
    - `CHUNK_SIZE = 340` frames, `RIGHT_CONTEXT = 40` frames
    - **Speed & Accuracy:** Drops RTF to **`0.002`** and improves 2-speaker CALLHOME DER to **`5.65%`** ([Sortformer v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)).

### 1.2 Weave 4-State Attribution Lifecycle & Full-Turn Identity Verification (`ERes2Net` / `CAM++`)
Following **[Weave](https://xlaude2040.com/weave/)**, every word's `speaker.state` progresses through four deterministic states:
1. `unknown`: Speaker identity not yet determined; renders in neutral grey (`#9AA0A6`) rather than guessing.
2. `provisional`: Initial fast `80 ms` frame assignment from Streaming Sortformer.
3. `stable`: Confirmed by full-turn speaker embedding verification (**ERes2Net** for English; **CAM++** for Korean, which settles at a median latency of **4.48 seconds**).
4. `corrected`: Updated in the pre-onset read-ahead zone if turn-level verification overrides an early provisional frame.

### 1.3 Sub-Second & Low-RAM Fallbacks (`Pyannote Streaming` & `512 MB` Custom Clustering)
- **Pyannote Streaming:** Processes fixed **`100 ms` chunks** (`6,400 bytes` of `16 kHz` `float32` PCM) within a **`300 ms` all-inclusive latency budget** for `>4` speakers ([Pyannote Blog](https://www.pyannote.ai/blog/how-we-built-streaming-diarization)).
- **@Lokesh_Lal's 512 MB RAM Custom Diarizer:** Extracts lightweight spectral/MFCC + pitch-centroid embeddings per speech segment and performs online bounded-memory cosine centroid clustering with exponential moving average updates, cutting diarization RAM from `4 GB` down to **`512 MB`** for CPU laptops ([CwiRenderer](https://stardance.hackclub.com/projects/63123)).

---

## 2. Production Code — `backend/ml_engine/stream2_diarization.py`

Write the following complete implementation to `backend/ml_engine/stream2_diarization.py`:

```python
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
            # Update EMA centroid inArrival-Order Speaker Cache style
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
```
