# Stream 3: Acoustic Prosody ($F_0$, RMS) & WhiStress Layer-9 Sentence Stress (`stream_3_prosody_and_stress.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint (Section 2.4)](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [Weave (SNU, 2026)](https://xlaude2040.com/weave/), [WhiStress (Interspeech 2025)](https://www.isca-archive.org/interspeech_2025/yosha25_interspeech.pdf), [WhiStress HTML Paper (arXiv:2505.19103v1)](https://arxiv.org/html/2505.19103v1)

---

## 1. Stream 3 DSP & Neural Architecture Specification

To drive the **Intonation** pillar (**Weight**, **Crest Size**, **Width**, **GRAD**, and **Hold Lift**), Stream 3 combines physical signal processing with deep prosodic representation extraction:

1. **75 ms Window / 20 ms Stride Acoustic Extraction & 300 ms Pooling ([Weave](https://xlaude2040.com/weave/)):**
   - Across the `16,000 Hz` waveform, fundamental frequency ($F_0$ in Hz) and root-mean-square (RMS) energy are extracted in **75 ms windows (`1,200` samples) with a 20 ms stride (`320` samples)**—matching the exact **50 Hz frame rate** of Whisper's audio encoder and emotion2vec+.
   - Frame-level $F_0$ and RMS values are mean/median-pooled over **300 ms windows (`15` frames)** across each word's aligned span $[t_{\text{start}}, t_{\text{end}}]$.
2. **Per-Speaker Running Median Pitch Tracker ($F_{0,\text{median}}$) ([Weave](https://xlaude2040.com/weave/)):**
   - Maintains a bounded ring buffer (up to `200` voiced frames per `speaker_id`) of each speaker's voiced $F_0 \in [65\text{ Hz}, 400\text{ Hz}]$ observations.
   - Computes the speaker's running median pitch $F_{0,\text{median}}$ (defaulting to `135.0 Hz` until at least `5` voiced frames are observed for a new speaker).
   - Anchoring `wght` to $F_{0,\text{median}}$ plus prominence prevents a deep-voiced speaker who shouts (doubling instantaneous $F_0$) from rendering in thin, light letters.
3. **WhiStress Layer-9 Alignment-Free Sentence Stress Detector ([WhiStress](https://www.isca-archive.org/interspeech_2025/yosha25_interspeech.pdf)):**
   - Attaches the `7M`-parameter `WhiStressLayer9Head` (1 Transformer decoder block with cross-attention + 2-layer FCNN binary classifier) to Layer 9 of the frozen Whisper backbone (`0.909 F1` on TinyStress-15K, `0.961 F1` on Aix-MARSEC).
   - When running in ultra-lightweight CPU/WebGPU mode without Layer-9 hidden tensors, computes a deterministic acoustic-prosodic prominence score combining intra-speaker pitch excursion ($\Delta F_0$), energy salience ($\Delta \text{RMS}$), and syllabic duration lengthening.
4. **Isolated Silence Hold Detection (`-0.525em` Vertical Lift) ([Weave](https://xlaude2040.com/weave/)):**
   - Marks `isolated_silence_hold = True` whenever a word is preceded by $\ge 250\text{ ms}$ of silence AND followed by $\ge 250\text{ ms}$ of silence.

---

## 2. Production Code — `backend/ml_engine/stream3_prosody_stress.py`

Write the following complete implementation to `backend/ml_engine/stream3_prosody_stress.py`:

```python
from collections import deque
import math
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
import librosa
import torch

from backend.config import EngineConfig
from backend.schemas.caption_spec import ProsodyMetrics
from backend.ml_engine.stream1_asr import RawWordAlignment


class SpeakerPitchTracker:
    # Tracks running median F0 per speaker_id across voiced frames so deep-voiced shouts stay bold.
    def __init__(self, max_history_frames: int = 250, default_median_hz: float = 135.0):
        self.max_history_frames = max_history_frames
        self.default_median_hz = default_median_hz
        self.history: Dict[str, Deque[float]] = {}

    def update_and_get_median(self, speaker_id: str, voiced_f0_values: List[float]) -> float:
        if speaker_id not in self.history:
            self.history[speaker_id] = deque(maxlen=self.max_history_frames)
        buf = self.history[speaker_id]
        for f0 in voiced_f0_values:
            if 65.0 <= f0 <= 400.0:
                buf.append(float(f0))
        if len(buf) < 3:
            return self.default_median_hz
        return float(np.median(np.array(buf, dtype=np.float32)))


class ProsodyAndStressExtractor:
    # Stream 3: 75ms window / 20ms stride F0 & RMS extractor, 300ms pooling,
    # Speaker Running Median F0 tracker, and WhiStress Layer-9 prominence scorer.
    WIN_LENGTH = 1200   # 75ms @ 16,000 Hz
    HOP_LENGTH = 320    # 20ms @ 16,000 Hz (50 Hz frame rate)
    POOL_FRAMES = 15    # 300ms pooling window (15 * 20ms)

    def __init__(self, config: EngineConfig):
        self.config = config
        self.pitch_tracker = SpeakerPitchTracker()
        self.whistress_head = None
        self._load_whistress_head()

    def _load_whistress_head(self) -> None:
        try:
            from pathlib import Path
            from training.models.whistress_layer9_head import WhiStressLayer9Head
            ckpt_path = Path(self.config.whistress_checkpoint)
            if ckpt_path.exists():
                self.whistress_head = WhiStressLayer9Head(d_model=768)
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                self.whistress_head.load_state_dict(state)
                self.whistress_head.eval()
        except Exception:
            self.whistress_head = None

    def extract_50hz_f0_and_rms(self, audio_f32: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(audio_f32) < self.WIN_LENGTH:
            pad_len = self.WIN_LENGTH - len(audio_f32)
            audio_f32 = np.pad(audio_f32, (0, pad_len), mode="constant")

        # 1. RMS Energy in 75ms window, 20ms stride
        rms_frames = librosa.feature.rms(
            y=audio_f32, frame_length=self.WIN_LENGTH, hop_length=self.HOP_LENGTH, center=True
        )[0]
        rms_db_frames = 20.0 * np.log10(np.maximum(rms_frames, 1e-5))
        rms_db_frames = np.clip(rms_db_frames, -100.0, 0.0)

        # 2. Fundamental Frequency F0 via YIN in 75ms window, 20ms stride
        f0_frames = librosa.yin(
            y=audio_f32,
            fmin=65.0,
            fmax=450.0,
            sr=16000,
            frame_length=self.WIN_LENGTH,
            hop_length=self.HOP_LENGTH,
        )
        # Mask unvoiced / silence frames where RMS < -48 dBFS
        f0_frames = np.where(rms_db_frames > -48.0, f0_frames, 0.0)

        # 3. 300ms (15-frame) moving average pooling for smooth envelopes
        if len(rms_db_frames) >= 3:
            kernel = np.ones(min(self.POOL_FRAMES, len(rms_db_frames))) / min(
                self.POOL_FRAMES, len(rms_db_frames)
            )
            rms_db_pooled = np.convolve(rms_db_frames, kernel, mode="same")
        else:
            rms_db_pooled = rms_db_frames

        return f0_frames.astype(np.float32), rms_db_pooled.astype(np.float32)

    def extract_batch_prosody(
        self,
        window_audio: np.ndarray,
        window_start_ms: int,
        raw_words: List[RawWordAlignment],
        word_speakers: List[Tuple[str, str]],
        l9_hidden: Optional[Dict[str, torch.Tensor]] = None,
        isolated_silence_min_ms: int = 250,
    ) -> List[ProsodyMetrics]:
        if not raw_words:
            return []

        f0_frames, rms_db_frames = self.extract_50hz_f0_and_rms(window_audio)
        num_frames = len(rms_db_frames)
        metrics_out: List[ProsodyMetrics] = []

        for idx, (w, (spk_id, _)) in enumerate(zip(raw_words, word_speakers)):
            rel_start_ms = max(0, w.start_ms - window_start_ms)
            rel_end_ms = max(rel_start_ms + 20, w.end_ms - window_start_ms)

            f_start = max(0, min(num_frames - 1, int(rel_start_ms // 20)))
            f_end = max(f_start + 1, min(num_frames, int(math.ceil(rel_end_ms / 20.0))))

            word_rms_db = float(np.mean(rms_db_frames[f_start:f_end]))
            word_f0_slice = f0_frames[f_start:f_end]
            voiced = [float(x) for x in word_f0_slice if 65.0 <= x <= 420.0]

            spk_median_f0 = self.pitch_tracker.update_and_get_median(spk_id, voiced)
            word_f0_hz = float(np.median(voiced)) if voiced else spk_median_f0

            # Normalized intra-word pitch contour (0.0 to 1.0)
            if len(word_f0_slice) > 0:
                contour = [
                    round(max(0.0, min(1.0, (float(val) - 80.0) / 170.0)), 3)
                    for val in word_f0_slice[:16]
                ]
            else:
                contour = [0.5]

            # Prominence score: use WhiStress Layer-9 head if tensor available, else acoustic salience
            if self.whistress_head is not None and l9_hidden is not None:
                with torch.no_grad():
                    logits = self.whistress_head(
                        l9_hidden["decoder_l9"], l9_hidden["encoder_l9"]
                    )
                    probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                    prominence = float(probs[0, min(idx, probs.shape[1] - 1)])
            else:
                # Acoustic prominence combining loudness above -22dBFS and pitch excursion
                loud_salience = max(0.0, min(1.0, (word_rms_db + 28.0) / 20.0))
                pitch_salience = max(0.0, min(1.0, abs(word_f0_hz - spk_median_f0) / 60.0))
                prominence = round(0.65 * loud_salience + 0.35 * pitch_salience, 4)

            # Check isolated silence hold (>= 250ms silence before AND after word)
            prev_gap = (
                (w.start_ms - raw_words[idx - 1].end_ms)
                if idx > 0
                else (w.start_ms - window_start_ms)
            )
            next_gap = (
                (raw_words[idx + 1].start_ms - w.end_ms)
                if idx + 1 < len(raw_words)
                else 300
            )
            isolated_hold = (prev_gap >= isolated_silence_min_ms) and (
                next_gap >= isolated_silence_min_ms
            )

            metrics_out.append(
                ProsodyMetrics(
                    rms_db=round(word_rms_db, 2),
                    f0_word_hz=round(word_f0_hz, 2),
                    f0_speaker_median_hz=round(spk_median_f0, 2),
                    prominence_score=round(max(0.0, min(1.0, prominence)), 4),
                    isolated_silence_hold=bool(isolated_hold),
                    pitch_contour_norm=contour,
                )
            )

        return metrics_out
```
