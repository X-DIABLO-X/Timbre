import math
from typing import Optional, Tuple
import numpy as np
import scipy.signal
import torch


class AudioPreprocessor:
    # Stage 0: 16kHz mono float32 normalizer, Silero VAD + SILENCE_RMS gate, and SCNet/Demucs stem separator.
    def __init__(
        self,
        silence_rms_threshold: float = 0.008,
        vad_prob_threshold: float = 0.45,
        enable_stem_separation: bool = False,
    ):
        self.silence_rms_threshold = silence_rms_threshold
        self.vad_prob_threshold = vad_prob_threshold
        self.enable_stem_separation = enable_stem_separation
        self.vad_model = None
        self.separator_model = None
        self._load_vad()
        if self.enable_stem_separation:
            self._load_separator()

    def _load_vad(self) -> None:
        try:
            from silero_vad import load_silero_vad
            self.vad_model = load_silero_vad(onnx=True)
        except Exception:
            self.vad_model = None

    def _load_separator(self) -> None:
        try:
            from demucs.pretrained import get_model
            self.separator_model = get_model("htdemucs")
            self.separator_model.eval()
        except Exception:
            self.separator_model = None

    def normalize_to_16k_mono_f32(
        self, pcm_bytes: bytes, sample_rate: int = 16000, is_float32: bool = False
    ) -> np.ndarray:
        if is_float32:
            audio = np.frombuffer(pcm_bytes, dtype=np.float32).copy()
        else:
            audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio = audio_i16.astype(np.float32) / 32768.0

        if sample_rate != 16000 and len(audio) > 0:
            target_len = int(round(len(audio) * (16000.0 / float(sample_rate))))
            audio = scipy.signal.resample(audio, target_len).astype(np.float32)

        return np.clip(audio, -1.0, 1.0)

    def compute_rms_and_db(self, audio_f32: np.ndarray) -> Tuple[float, float]:
        if len(audio_f32) == 0:
            return 0.0, -100.0
        rms = float(np.sqrt(np.mean(np.square(audio_f32)) + 1e-12))
        rms_db = max(-100.0, min(0.0, 20.0 * math.log10(max(rms, 1e-5))))
        return rms, rms_db

    def run_silero_vad(self, audio_f32: np.ndarray) -> bool:
        rms, _ = self.compute_rms_and_db(audio_f32)
        if rms < self.silence_rms_threshold:
            return False

        if self.vad_model is None:
            return rms >= self.silence_rms_threshold

        # Silero VAD expects 512-sample chunks at 16,000 Hz (32ms windows)
        win_size = 512
        max_prob = 0.0
        with torch.no_grad():
            for i in range(0, len(audio_f32) - win_size + 1, win_size):
                chunk_t = torch.from_numpy(audio_f32[i : i + win_size])
                prob = float(self.vad_model(chunk_t, 16000).item())
                if prob > max_prob:
                    max_prob = prob
        return max_prob >= self.vad_prob_threshold

    def separate_stems(self, audio_f32: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.enable_stem_separation or self.separator_model is None or len(audio_f32) < 1600:
            return audio_f32, None

        # Run Demucs/SCNet separation and resample back to 16kHz mono
        try:
            from demucs.apply import apply_model
            with torch.no_grad():
                wav_44k = scipy.signal.resample(audio_f32, int(len(audio_f32) * 44100 / 16000))
                wav_t = torch.from_numpy(wav_44k).float().unsqueeze(0).repeat(2, 1).unsqueeze(0)
                sources = apply_model(self.separator_model, wav_t, shifts=0, split=True)[0]
                # htdemucs order: drums(0), bass(1), other(2), vocals(3)
                vocals_44k = sources[3].mean(dim=0).cpu().numpy()
                bgm_44k = (sources[0] + sources[1] + sources[2]).mean(dim=0).cpu().numpy()

                vocals_16k = scipy.signal.resample(vocals_44k, len(audio_f32)).astype(np.float32)
                bgm_16k = scipy.signal.resample(bgm_44k, len(audio_f32)).astype(np.float32)
                return vocals_16k, bgm_16k
        except Exception:
            return audio_f32, None

    def process_chunk(
        self, pcm_bytes: bytes, sample_rate: int = 16000, is_float32: bool = False
    ) -> Tuple[np.ndarray, bool, float, Optional[np.ndarray]]:
        audio_f32 = self.normalize_to_16k_mono_f32(pcm_bytes, sample_rate, is_float32)
        _, rms_db = self.compute_rms_and_db(audio_f32)
        vocals_f32, bgm_stem = self.separate_stems(audio_f32)
        is_speech = self.run_silero_vad(vocals_f32)
        return vocals_f32, is_speech, rms_db, bgm_stem
