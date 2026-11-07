# Stream 4: Paralinguistic Event Detection, SFX & Vocal Affect (`stream_4_paralinguistics_affect.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint (Section 2.5)](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [FunAudioLLM SenseVoice](https://fun-audio-llm.github.io/), [emotion2vec+ Large](https://huggingface.co/emotion2vec/emotion2vec_plus_large), [emotion2vec-plus-large-mlx](https://huggingface.co/mlx-community/emotion2vec-plus-large-mlx/blob/main/README.md), [emotion2vec-web-distill](https://huggingface.co/thomashallock/emotion2vec-web-distill), [VocalSound & Nonspeech7k](https://digital-library.theiet.org/doi/full/10.1049/sil2.12233)

---

## 1. Stream 4 Architecture Specification

1. **SenseVoice-Small / Large (`iic/SenseVoiceSmall` & `iic/SenseVoiceLarge`):**
   - **SenseVoice-Small** is a non-autoregressive encoder-only foundation model that executes **~15× faster than Whisper-large-v3** ([FunAudioLLM](https://fun-audio-llm.github.io/)).
   - In a single forward pass, it outputs:
     - Speech Emotion Recognition (SER) tokens: `<|HAPPY|>`, `<|SAD|>`, `<|ANGRY|>`, `<|NEUTRAL|>`, `<|Unknown_Emo|>`.
     - Audio Event Detection (AED) tokens: **music (`<|BGM|>`), applause (`<|Applause|>`), laughter (`[laughter]`), crying (`[crying]`), coughing (`[coughing]`), sneezing (`[sneezing]`), and breathing (`[breathing]`)**.
   - When exact start and end boundary timestamps of `<|BGM|>` or `<|Applause|>` are required in offline VOD mode, **SenseVoice-Large** (encoder-decoder architecture) explicitly predicts event onset/offset timestamps ([FunAudioLLM](https://fun-audio-llm.github.io/)).
2. **emotion2vec+ (`iic/emotion2vec_plus_base` ~90M / `iic/emotion2vec_plus_large` ~300M / `thomashallock/emotion2vec-web-distill` 2.23M):**
   - A Data2Vec 2.0 foundation model (`conv feature extractor -> transformer encoder -> 9-class linear head`) classifying 9 discrete categories (*angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown*) and outputting **50 Hz (`20 ms` stride) frame embeddings** when `granularity="frame"` ([emotion2vec+ Large](https://huggingface.co/emotion2vec/emotion2vec_plus_large)).
   - Combined with our custom downstream multi-task head trained on **VocalSound** and **Nonspeech7k**, it detects fine-grained human vocalizations (`[sighs]`, `[gasps]`, `[screaming]`, `[yawning]`, `[throat clearing]`, `[sniffs]`) and continuous vocal strain driving Roboto Flex's `XOPQ`/`YOPQ` stroke contrast axes.

---

## 2. Production Code — `backend/ml_engine/stream4_paralinguistics.py`

Write the following complete implementation to `backend/ml_engine/stream4_paralinguistics.py`:

```python
import re
from typing import List, Optional, Tuple
import numpy as np

from backend.config import EngineConfig
from backend.schemas.caption_spec import NonSpeechEvent


SENSEVOICE_EVENT_MAP = {
    "<|BGM|>": ("BGM", "<|BGM|>"),
    "<|Music|>": ("BGM", "<|BGM|>"),
    "<|Applause|>": ("APPLAUSE", "<|Applause|>"),
    "<|Laughter|>": ("LAUGHTER", "[laughter]"),
    "<|Cry|>": ("CRYING", "[crying]"),
    "<|Cough|>": ("COUGH", "[coughing]"),
    "<|Sneeze|>": ("SNEEZE", "[sneezing]"),
    "<|Breath|>": ("BREATHING", "[breathing]"),
    "<|Sigh|>": ("SIGH", "[sighs]"),
    "<|Gasp|>": ("GASP", "[gasps]"),
}


class ParalinguisticEventEngine:
    # Stream 4: SenseVoice-Small/Large audio event tagging + emotion2vec+ 50Hz frame affect.
    def __init__(self, config: EngineConfig):
        self.config = config
        self.sensevoice_model = None
        self.emotion2vec_model = None
        self._load_models()

    def _load_models(self) -> None:
        try:
            from funasr import AutoModel
            self.sensevoice_model = AutoModel(
                model=self.config.sensevoice_model_id,
                trust_remote_code=True,
                device=self.config.device if self.config.device != "mps" else "cpu",
                disable_update=True,
            )
        except Exception:
            self.sensevoice_model = None

        try:
            from funasr import AutoModel
            if "mlx" not in self.config.emotion2vec_model_id:
                self.emotion2vec_model = AutoModel(
                    model=self.config.emotion2vec_model_id,
                    disable_update=True,
                )
        except Exception:
            self.emotion2vec_model = None

    def detect_events_and_affect(
        self,
        audio_f32: np.ndarray,
        bgm_stem: Optional[np.ndarray],
        window_start_ms: int,
    ) -> Tuple[List[NonSpeechEvent], np.ndarray]:
        dur_ms = max(100, int(round((len(audio_f32) / 16000.0) * 1000.0)))
        num_frames_50hz = max(1, dur_ms // 20)
        events_out: List[NonSpeechEvent] = []

        # 1. Check separated BGM stem energy if SCNet/Demucs is active
        if bgm_stem is not None and len(bgm_stem) > 0:
            bgm_rms = float(np.sqrt(np.mean(np.square(bgm_stem)) + 1e-12))
            if bgm_rms > 0.025:  # > -32 dBFS sustained music stem
                events_out.append(
                    NonSpeechEvent(
                        type="BGM",
                        label="<|BGM|>",
                        start_ms=window_start_ms,
                        end_ms=window_start_ms + dur_ms,
                        confidence=0.92,
                    )
                )

        # 2. Run SenseVoice-Small for <15x faster single-pass AED & SER tags
        if self.sensevoice_model is not None and len(audio_f32) >= 1600:
            try:
                res = self.sensevoice_model.generate(
                    input=audio_f32,
                    cache={},
                    language="auto",
                    use_itn=False,
                    batch_size_s=30,
                )
                raw_text = res[0].get("text", "") if res else ""
                for tag_str, (ev_type, ev_label) in SENSEVOICE_EVENT_MAP.items():
                    if tag_str in raw_text:
                        if not any(e.type == ev_type for e in events_out):
                            events_out.append(
                                NonSpeechEvent(
                                    type=ev_type,  # type: ignore
                                    label=ev_label,  # type: ignore
                                    start_ms=window_start_ms,
                                    end_ms=window_start_ms + dur_ms,
                                    confidence=0.95,
                                )
                            )
            except Exception:
                pass

        # 3. Extract 50Hz frame affect representations via emotion2vec+
        affect_frames = np.zeros((num_frames_50hz, 9), dtype=np.float32)
        affect_frames[:, 4] = 1.0  # Default neutral class = index 4

        if self.emotion2vec_model is not None and len(audio_f32) >= 1600:
            try:
                e2v_res = self.emotion2vec_model.generate(
                    audio_f32,
                    granularity="frame",
                    extract_embedding=True,
                )
                if e2v_res and "scores" in e2v_res[0]:
                    scores = np.array(e2v_res[0]["scores"], dtype=np.float32)
                    if scores.ndim == 2:
                        affect_frames = scores
            except Exception:
                pass

        return events_out, affect_frames
```
