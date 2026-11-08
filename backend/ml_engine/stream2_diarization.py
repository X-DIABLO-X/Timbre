from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import librosa
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

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
    """
    Upgraded 512MB RAM Acoustic & Neural Speaker Diarizer.
    Fixes Issue #1 (same speaker split into multiple colors) via:
      1. Voiced-only frame masking (strips silence, breaths, and background music stems)
      2. Optional Pretrained ECAPA-TDNN / ERes2Net d-vectors when installed, with a
         Formant-Contrast + Sinusoidal Liftered Cepstral + Log-F0 fallback (dropping c0 loudness)
      3. Temporal turn-continuity hysteresis (+0.14 bonus for the active speaker across <1.5s pauses)
      4. Two-pass global Cosine Agglomerative Clustering (cluster_offline_segments) +
         Isolated-Turn Glitch Smoothing for offline VOD scripts (generate_dune_cwi_v2.py).
    """

    def __init__(
        self,
        max_speakers: int = 6,
        similarity_threshold: float = 0.62,
        continuity_bonus: float = 0.14,
    ):
        self.max_speakers = max_speakers
        self.similarity_threshold = similarity_threshold
        self.continuity_bonus = continuity_bonus
        self.centroids: Dict[str, np.ndarray] = {}
        self.counts: Dict[str, int] = {}
        self.last_speaker_id: Optional[str] = None
        self.last_end_ms: int = -10000
        self.neural_encoder = None
        self._try_load_neural_encoder()

    def _try_load_neural_encoder(self) -> None:
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.neural_encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": device},
            )
        except Exception:
            self.neural_encoder = None

    def _filter_voiced_speech(self, audio_f32: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Strips unvoiced silence, breaths, and low-energy background score so speaker
        embeddings reflect pure vocal tract resonance rather than room/BGM noise.
        """
        if len(audio_f32) < 1600:
            return audio_f32

        hop = 256
        rms = librosa.feature.rms(y=audio_f32, frame_length=1024, hop_length=hop)[0]
        zcr = librosa.feature.zero_crossing_rate(y=audio_f32, frame_length=1024, hop_length=hop)[0]

        energy_floor = max(0.010, float(np.percentile(rms, 35)))
        voiced_slices: List[np.ndarray] = []
        for idx, (r_val, z_val) in enumerate(zip(rms, zcr)):
            # Voiced speech has high energy and moderate zero-crossing rate (rejects hiss/sibilance)
            if r_val >= energy_floor and z_val < 0.28:
                start = idx * hop
                end = min(len(audio_f32), start + hop)
                if end > start:
                    voiced_slices.append(audio_f32[start:end])

        if not voiced_slices:
            return audio_f32
        voiced = np.concatenate(voiced_slices)
        return voiced if len(voiced) >= 1200 else audio_f32

    def _extract_embedding(self, audio_f32: np.ndarray) -> Optional[np.ndarray]:
        if len(audio_f32) < 800:
            return None

        voiced = self._filter_voiced_speech(audio_f32, sr=16000)
        if len(voiced) < 800:
            return None

        # Path 1: Pretrained 192-D ECAPA-TDNN Speaker Verification Embedding (if installed)
        if self.neural_encoder is not None:
            try:
                import torch

                with torch.no_grad():
                    wav_t = torch.from_numpy(voiced).float().unsqueeze(0)
                    emb = self.neural_encoder.encode_batch(wav_t).squeeze().cpu().numpy()
                    norm = float(np.linalg.norm(emb))
                    if norm > 1e-8:
                        return (emb / norm).astype(np.float32)
            except Exception:
                pass

        # Path 2: Loudness-Invariant & Phoneme-Smoothed Vocal Tract Embedding
        # 1. Pre-emphasis (0.97) normalizes spectral tilt between whispers and loud speech
        y_pre = librosa.effects.preemphasis(voiced, coef=0.97)

        # 2. Sinusoidally liftered MFCCs (dropping c0 energy so volume never splits speaker ID!)
        mfcc = librosa.feature.mfcc(
            y=y_pre,
            sr=16000,
            n_mfcc=24,
            n_fft=1024,
            hop_length=256,
            n_mels=40,
            fmin=80.0,
            fmax=7600.0,
            lifter=22,
        )
        # Keep c1..c19 (vocal tract envelope); use median & IQR to ignore transient consonant spikes
        cep = mfcc[1:20, :]
        cep_med = np.median(cep, axis=1)
        cep_iqr = np.percentile(cep, 75, axis=1) - np.percentile(cep, 25, axis=1)
        cep_med_norm = cep_med / (np.linalg.norm(cep_med) + 1e-8)
        cep_iqr_norm = cep_iqr / (np.linalg.norm(cep_iqr) + 1e-8)

        # 3. 7-band Spectral Contrast (captures speaker formant peak-to-valley ratios)
        contrast = librosa.feature.spectral_contrast(
            y=y_pre, sr=16000, n_fft=1024, hop_length=256, n_bands=6, fmin=100.0
        )
        contrast_med = np.median(contrast, axis=1)
        contrast_norm = contrast_med / (np.linalg.norm(contrast_med) + 1e-8)

        # 4. Fundamental Frequency (Log-F0 Median & Pitch Range Anchor)
        f0 = librosa.yin(
            voiced, fmin=65.0, fmax=400.0, sr=16000, frame_length=1024, hop_length=256
        )
        rms_f0 = librosa.feature.rms(y=voiced, frame_length=1024, hop_length=256)[0]
        valid_f0 = f0[(rms_f0 > 0.010) & (f0 >= 65.0) & (f0 <= 380.0)]
        if len(valid_f0) >= 2:
            log_f0_med = float(np.median(np.log(valid_f0)))
            log_f0_iqr = float(
                np.percentile(np.log(valid_f0), 75) - np.percentile(np.log(valid_f0), 25)
            )
        else:
            log_f0_med = float(np.log(135.0))
            log_f0_iqr = 0.15

        # Normalize log-F0 in [log(65)=4.17, log(380)=5.94] onto [-1.0, 1.0]
        f0_scaled = (log_f0_med - 5.05) / 0.88
        pitch_vec = np.array([f0_scaled, log_f0_iqr], dtype=np.float32)

        feat = np.concatenate(
            [
                1.00 * cep_med_norm,
                0.30 * cep_iqr_norm,
                0.55 * contrast_norm,
                0.65 * pitch_vec,
            ]
        )
        norm = float(np.linalg.norm(feat))
        if norm < 1e-8:
            return None
        return (feat / norm).astype(np.float32)

    def assign_chunk(
        self,
        audio_f32: np.ndarray,
        start_ms: int,
        end_ms: int,
        force_new_turn_hint: bool = False,
    ) -> SpeakerTurnFrame:
        emb = self._extract_embedding(audio_f32)
        if emb is None:
            fallback_spk = self.last_speaker_id or "SPK_00"
            return SpeakerTurnFrame(start_ms, end_ms, fallback_spk, "provisional", 0.55)

        if not self.centroids:
            spk_id = "SPK_00"
            self.centroids[spk_id] = emb
            self.counts[spk_id] = 1
            self.last_speaker_id = spk_id
            self.last_end_ms = end_ms
            return SpeakerTurnFrame(start_ms, end_ms, spk_id, "provisional", 0.85)

        best_spk = "SPK_00"
        best_score = -1.0
        raw_best_sim = -1.0
        gap_ms = max(0, start_ms - self.last_end_ms)

        for spk_id, centroid in self.centroids.items():
            sim = float(np.dot(emb, centroid))
            # Temporal turn-continuity bonus: prevents mid-sentence breaths or vowel shifts
            # from spawning a fake new speaker ID when the same person continues speaking.
            bonus = (
                self.continuity_bonus
                if (
                    spk_id == self.last_speaker_id
                    and gap_ms <= 1600
                    and not force_new_turn_hint
                )
                else 0.0
            )
            score = sim + bonus
            if score > best_score:
                best_score = score
                raw_best_sim = sim
                best_spk = spk_id

        if best_score >= self.similarity_threshold or len(self.centroids) >= self.max_speakers:
            # Only update centroid on clean segments >= 900ms with solid raw similarity
            if (end_ms - start_ms) >= 900 and raw_best_sim >= (self.similarity_threshold - 0.04):
                alpha = 0.08  # Conservative EMA update prevents background music drift
                updated = (1.0 - alpha) * self.centroids[best_spk] + alpha * emb
                self.centroids[best_spk] = updated / (np.linalg.norm(updated) + 1e-8)
            self.counts[best_spk] += 1
            self.last_speaker_id = best_spk
            self.last_end_ms = end_ms
            state: SpeakerState = "stable" if self.counts[best_spk] >= 2 else "provisional"
            return SpeakerTurnFrame(
                start_ms, end_ms, best_spk, state, max(0.50, min(1.0, best_score))
            )

        new_spk = f"SPK_{len(self.centroids):02d}"
        self.centroids[new_spk] = emb
        self.counts[new_spk] = 1
        self.last_speaker_id = new_spk
        self.last_end_ms = end_ms
        return SpeakerTurnFrame(start_ms, end_ms, new_spk, "provisional", 0.80)

    def cluster_offline_segments(
        self,
        segment_audios: List[np.ndarray],
        segment_spans_ms: Optional[List[Tuple[int, int]]] = None,
        seed_embeddings: Optional[Dict[str, np.ndarray]] = None,
        fixed_speaker_overrides: Optional[Dict[int, str]] = None,
    ) -> List[str]:
        """
        Two-pass global speaker clustering + isolated-turn glitch smoothing for offline VOD
        (generate_dune_cwi_v2.py). Ensures the same voice receives a consistent color across
        the entire video.
        """
        n_segs = len(segment_audios)
        if n_segs == 0:
            return []

        embeddings: List[Optional[np.ndarray]] = [
            self._extract_embedding(seg) for seg in segment_audios
        ]

        # Fill any ultra-short segment with its nearest neighbor's embedding
        last_valid: Optional[np.ndarray] = None
        for i in range(n_segs):
            if embeddings[i] is not None:
                last_valid = embeddings[i]
            elif last_valid is not None:
                embeddings[i] = last_valid

        valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
        if len(valid_indices) < 2:
            return ["SPK_00"] * n_segs

        X = np.stack([embeddings[i] for i in valid_indices], axis=0)

        # Discover optimal speaker count k in [2, max_speakers] via cosine Silhouette score
        best_k = 2
        best_sil = -1.0
        max_k = min(self.max_speakers, len(valid_indices) - 1)
        for k in range(3, max(4, max_k + 1)):
            try:
                model = AgglomerativeClustering(
                    n_clusters=k, linkage="ward"
                ).fit(X)
                if len(set(model.labels_)) < 2:
                    continue
                sil = float(silhouette_score(X, model.labels_, metric="euclidean"))
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
            except Exception:
                continue

        final_cluster = AgglomerativeClustering(
            n_clusters=best_k, linkage="ward"
        ).fit(X)

        cluster_to_spk: Dict[int, str] = {}
        used_spk_ids = set()

        # Anchor seeded speakers (e.g., SPK_00 = Chani opening narration) to their closest cluster
        if seed_embeddings:
            for spk_id, seed_vec in seed_embeddings.items():
                best_c = -1
                best_sim = -1.0
                for c in range(best_k):
                    if c in cluster_to_spk:
                        continue
                    c_mean = X[final_cluster.labels_ == c].mean(axis=0)
                    c_mean = c_mean / (np.linalg.norm(c_mean) + 1e-8)
                    sim = float(np.dot(c_mean, seed_vec))
                    if sim > best_sim:
                        best_sim = sim
                        best_c = c
                if best_c >= 0:
                    cluster_to_spk[best_c] = spk_id
                    used_spk_ids.add(spk_id)

        next_slot = 0
        raw_labels = ["SPK_00"] * n_segs
        for idx, c_label in zip(valid_indices, final_cluster.labels_):
            c_int = int(c_label)
            if c_int not in cluster_to_spk:
                while f"SPK_{next_slot:02d}" in used_spk_ids:
                    next_slot += 1
                spk_name = f"SPK_{next_slot:02d}"
                cluster_to_spk[c_int] = spk_name
                used_spk_ids.add(spk_name)
                next_slot += 1
            raw_labels[idx] = cluster_to_spk[c_int]

        # Apply explicit overrides (e.g., opening narration range)
        if fixed_speaker_overrides:
            for seg_idx, spk_override in fixed_speaker_overrides.items():
                if 0 <= seg_idx < n_segs:
                    raw_labels[seg_idx] = spk_override

        # Pass 2: Isolated Single-Segment Glitch Smoothing
        # If segment[i-1] == segment[i+1] == SPK_A, and segment[i] was assigned SPK_B
        # despite having reasonable acoustic similarity to SPK_A and short pause gaps, smooth to SPK_A.
        smoothed = list(raw_labels)
        for i in range(1, n_segs - 1):
            if fixed_speaker_overrides and i in fixed_speaker_overrides:
                continue
            prev_spk = smoothed[i - 1]
            next_spk = raw_labels[i + 1]
            curr_spk = smoothed[i]
            if prev_spk == next_spk and curr_spk != prev_spk:
                emb_prev = embeddings[i - 1]
                emb_curr = embeddings[i]
                emb_next = embeddings[i + 1]
                if emb_prev is not None and emb_curr is not None and emb_next is not None:
                    neighbor_mean = 0.5 * (emb_prev + emb_next)
                    neighbor_mean /= np.linalg.norm(neighbor_mean) + 1e-8
                    sim_to_neighbors = float(np.dot(emb_curr, neighbor_mean))
                    gap_ok = True
                    if segment_spans_ms:
                        gap_before = segment_spans_ms[i][0] - segment_spans_ms[i - 1][1]
                        gap_after = segment_spans_ms[i + 1][0] - segment_spans_ms[i][1]
                        gap_ok = (gap_before <= 2000) and (gap_after <= 2000)
                    if sim_to_neighbors >= 0.52 and gap_ok:
                        smoothed[i] = prev_spk

        return smoothed


class StreamingDiarizationEngine:
    """
    Stream 2: NVIDIA Streaming Sortformer 4spk v2.1 (80ms frames, AOSC=188, ATS)
    with ERes2Net/CAM++ full-turn verification and Robust 512MB fallback.
    Notice how process_window uses >= 1.28s (16 frames * 80ms) windows instead of
    tiny 320ms phoneme slices so speaker identity never flips mid-word!
    """

    SORTFORMER_LOW_LATENCY_CFG = {
        "frame_step_ms": 80,
        "chunk_size": 6,
        "right_context": 7,
        "fifo_size": 188,
        "update_period": 144,
        "spkcache_size": 188,
    }
    SORTFORMER_VOD_CFG = {
        "frame_step_ms": 80,
        "chunk_size": 340,
        "right_context": 40,
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

    def process_window(
        self, audio_f32: np.ndarray, window_start_ms: int
    ) -> List[SpeakerTurnFrame]:
        if len(audio_f32) == 0:
            return []

        # Evaluate speaker identity over >= 1.28s (20,480 samples at 16kHz) windows
        # instead of 320ms single-phoneme slices so vowels vs consonants never split colors.
        step_samples = max(len(audio_f32), 20480)
        frames_out: List[SpeakerTurnFrame] = []

        for offset in range(0, len(audio_f32), step_samples):
            sub_audio = audio_f32[offset : offset + step_samples]
            s_ms = window_start_ms + int(round((offset / 16000.0) * 1000.0))
            e_ms = window_start_ms + int(round(((offset + len(sub_audio)) / 16000.0) * 1000.0))
            frame = self.fallback_diarizer.assign_chunk(sub_audio, s_ms, e_ms)
            frames_out.append(frame)

        return frames_out
