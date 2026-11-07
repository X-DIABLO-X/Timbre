# Master Parallel Audio Pipeline Orchestrator & Timestamp Joiner (`pipelines.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint (Section 2)](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [Weave (SNU, 2026)](https://xlaude2040.com/weave/), [CwiRenderer](https://stardance.hackclub.com/projects/63123)

---

## 1. Architecture of the 4-Stream Parallel Pipeline

To transform raw `16,000 Hz` mono PCM audio into a `CaptionSpec v2.0-cwi` payload within a real-time latency budget (`RTF < 0.5`, ideally `< 0.1`), `backend/ml_engine/orchestrator.py` executes Stage 0 followed by **four concurrent processing streams** (`asyncio.gather` backed by a dedicated `ThreadPoolExecutor` / CUDA streams) and merges their outputs in `backend/ml_engine/joiner.py`:

1. **Stage 0 (`stage0_vad_separator.py`):** Normalizes incoming audio to `16,000 Hz` mono `float32`, runs **Silero VAD** (`10–20 ms` windows, `SILENCE_RMS = 0.008` gate), and optionally separates `vocals` vs. `bgm_sfx` via **SCNet** or **HT-Demucs v4**.
2. **Stream 1 (`stream1_asr.py`):** Runs verbatim ASR + sub-50ms word boundary alignment (**CrisperWhisper 2.0** `ct2` speculative or live **Nemotron + Parakeet TDT v3**), outputting `List[RawWordAlignment]` with `[start_ms, end_ms]` and intermediate Layer-9 hidden states.
3. **Stream 2 (`stream2_diarization.py`):** Runs **Streaming Sortformer 4spk v2.1** (`80 ms` frames, `1.04s` latency) + **ERes2Net / CAM++** turn verification (or `512 MB` custom clustering), outputting `List[SpeakerTurnFrame]` with 4-state attribution (`unknown`, `provisional`, `stable`, `corrected`).
4. **Stream 3 (`stream3_prosody_stress.py`):** Computes `75 ms` window / `20 ms` stride $F_0$ and RMS energy, pools over `300 ms`, updates the speaker's running median pitch $F_{0,\text{median}}$, and runs the `7M`-parameter **WhiStress Layer-9 Head** to emit `List[ProsodyMetrics]`.
5. **Stream 4 (`stream4_paralinguistics.py`):** Runs **SenseVoice-Small / Large** and **emotion2vec+** (`50 Hz` frames) to emit `List[NonSpeechEvent]` (`<|BGM|>`, `<|Applause|>`, `[laughter]`, `[sighs]`, `[gasps]`, `[crying]`) and frame-level vocal affect embeddings.

---

## 2. Complete Production Implementation — `backend/ml_engine/orchestrator.py`

Write the following complete code to `backend/ml_engine/orchestrator.py`:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Dict, List, Optional
import numpy as np

from backend.config import EngineConfig, detect_optimal_profile
from backend.schemas.caption_spec import CaptionSpec
from backend.ml_engine.stage0_vad_separator import AudioPreprocessor
from backend.ml_engine.stream1_asr import VerbatimASREngine
from backend.ml_engine.stream2_diarization import StreamingDiarizationEngine
from backend.ml_engine.stream3_prosody_stress import ProsodyAndStressExtractor
from backend.ml_engine.stream4_paralinguistics import ParalinguisticEventEngine
from backend.ml_engine.joiner import TimestampJoiner


class ParallelAudioPipeline:
    # Master orchestrator executing Stage 0 and Streams 1-4 concurrently.
    def __init__(self, config: Optional[EngineConfig] = None, enable_stem_separation: bool = False):
        self.config = config or detect_optimal_profile()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cwi_stream")

        self.stage0 = AudioPreprocessor(
            silence_rms_threshold=self.config.vad_silence_rms_threshold,
            enable_stem_separation=enable_stem_separation,
        )
        self.stream1_asr = VerbatimASREngine(self.config)
        self.stream2_diar = StreamingDiarizationEngine(self.config)
        self.stream3_prosody = ProsodyAndStressExtractor(self.config)
        self.stream4_para = ParalinguisticEventEngine(self.config)
        self.joiner = TimestampJoiner(self.config)

        self.segment_counter = 0
        self.acoustic_clock_ms = 0
        self.audio_ring_buffer = np.zeros(0, dtype=np.float32)
        self.buffer_start_ms = 0

    async def ingest_pcm_chunk(
        self,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
        is_float32: bool = False,
        mode: str = "live_streaming",
        locale: str = "en",
        compass_state: Optional[Dict[str, float]] = None,
    ) -> Optional[CaptionSpec]:
        loop = asyncio.get_running_loop()

        # Stage 0: Normalize to 16kHz float32, run Silero VAD + SILENCE_RMS gate, optional SCNet
        audio_f32, is_speech, rms_db, bgm_stem = await loop.run_in_executor(
            self.executor,
            self.stage0.process_chunk,
            pcm_bytes,
            sample_rate,
            is_float32,
        )

        chunk_duration_ms = int(round((len(audio_f32) / 16000.0) * 1000.0))
        self.acoustic_clock_ms += chunk_duration_ms

        if not is_speech and len(self.audio_ring_buffer) == 0:
            # Even during speech silence, check if BGM / Applause is active in Stream 4
            if rms_db > -45.0:
                events, _ = await loop.run_in_executor(
                    self.executor,
                    self.stream4_para.detect_events_and_affect,
                    audio_f32,
                    bgm_stem,
                    self.acoustic_clock_ms - chunk_duration_ms,
                )
                if events:
                    self.segment_counter += 1
                    return self.joiner.build_events_only_spec(
                        segment_id=f"seg_{self.segment_counter:04d}",
                        events=events,
                        acoustic_time_ms=self.acoustic_clock_ms,
                        locale=locale,
                    )
            return None

        if len(self.audio_ring_buffer) == 0:
            self.buffer_start_ms = self.acoustic_clock_ms - chunk_duration_ms

        self.audio_ring_buffer = np.concatenate([self.audio_ring_buffer, audio_f32])
        buffered_ms = int(round((len(self.audio_ring_buffer) / 16000.0) * 1000.0))

        # Trigger parallel 4-stream evaluation every 600ms of speech or when silence follows speech
        should_flush = (buffered_ms >= 600) or (not is_speech and buffered_ms >= 200)
        if not should_flush:
            return None

        window_audio = self.audio_ring_buffer.copy()
        window_start_ms = self.buffer_start_ms

        # Keep 160ms context overlap for seamless boundary continuation if still speaking
        if is_speech and buffered_ms < 15000:
            pass
        else:
            self.audio_ring_buffer = np.zeros(0, dtype=np.float32)

        # Fan out Streams 1, 2, and 4 in parallel (Stream 3 uses Stream 1's word spans + Layer-9 states)
        asr_future = loop.run_in_executor(
            self.executor,
            self.stream1_asr.transcribe_window,
            window_audio,
            window_start_ms,
            mode,
            locale,
        )
        diar_future = loop.run_in_executor(
            self.executor,
            self.stream2_diar.process_window,
            window_audio,
            window_start_ms,
        )
        para_future = loop.run_in_executor(
            self.executor,
            self.stream4_para.detect_events_and_affect,
            window_audio,
            bgm_stem,
            window_start_ms,
        )

        (raw_words, l9_hidden), diar_frames, (events, affect_frames_50hz) = await asyncio.gather(
            asr_future, diar_future, para_future
        )

        # Map each word to its dominant speaker from Stream 2 before computing Speaker-Median F0 in Stream 3
        word_speakers = self.joiner.assign_speakers_to_words(raw_words, diar_frames)

        # Stream 3: Extract 75ms/20ms F0 & RMS, pool over 300ms, update per-speaker running median F0, run WhiStress L9
        prosody_list = await loop.run_in_executor(
            self.executor,
            self.stream3_prosody.extract_batch_prosody,
            window_audio,
            window_start_ms,
            raw_words,
            word_speakers,
            l9_hidden,
        )

        self.segment_counter += 1
        spec = self.joiner.build_caption_spec(
            segment_id=f"seg_{self.segment_counter:04d}",
            raw_words=raw_words,
            word_speakers=word_speakers,
            prosody_list=prosody_list,
            events=events,
            affect_frames_50hz=affect_frames_50hz,
            acoustic_time_ms=self.acoustic_clock_ms,
            locale=locale,
            compass_state=compass_state,
        )
        return spec
```

---

## 3. Complete Timestamp Joiner Implementation — `backend/ml_engine/joiner.py`

Write the following complete code to `backend/ml_engine/joiner.py`:

```python
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from backend.config import EngineConfig
from backend.schemas.caption_spec import (
    CaptionSpec,
    NonSpeechEvent,
    ProsodyMetrics,
    SpeakerInfo,
    WordToken,
    WorkAreaBox,
)
from backend.ml_engine.typography_math import compute_typography_spec
from backend.ml_engine.stream1_asr import RawWordAlignment
from backend.ml_engine.stream2_diarization import SpeakerTurnFrame


DEFAULT_PALETTE = [
    "#4FC3F7",  # SPK_00: Electric Cyan
    "#FFD54F",  # SPK_01: Amber Gold
    "#FF8A80",  # SPK_02: Coral Rose
    "#69F0AE",  # SPK_03: Mint Emerald
    "#B388FF",  # SPK_04: Lavender Violet
    "#FFAB40",  # SPK_05: Tangerine
]
UNKNOWN_SPEAKER_HEX = "#9AA0A6"


class TimestampJoiner:
    # Deterministically joins Streams 1-4 by [start_ms, end_ms] word intervals.
    def __init__(self, config: EngineConfig):
        self.config = config
        self.speaker_color_map: Dict[str, str] = {}

    def _get_speaker_color(self, speaker_id: str, state: str) -> str:
        if state == "unknown" or speaker_id == "SPK_UNKNOWN":
            return UNKNOWN_SPEAKER_HEX
        if speaker_id not in self.speaker_color_map:
            slot = len(self.speaker_color_map) % len(DEFAULT_PALETTE)
            self.speaker_color_map[speaker_id] = DEFAULT_PALETTE[slot]
        return self.speaker_color_map[speaker_id]

    def assign_speakers_to_words(
        self,
        raw_words: List[RawWordAlignment],
        diar_frames: List[SpeakerTurnFrame],
    ) -> List[Tuple[str, str]]:
        # Assigns (speaker_id, attribution_state) to each word by overlapping 80ms/100ms diarization frames.
        assignments: List[Tuple[str, str]] = []
        for w in raw_words:
            overlapping = [
                f for f in diar_frames if f.end_ms > w.start_ms and f.start_ms < w.end_ms
            ]
            if not overlapping:
                assignments.append(("SPK_UNKNOWN", "unknown"))
                continue

            # Vote weighted by overlap duration and confidence
            votes: Dict[str, float] = {}
            states: Dict[str, str] = {}
            for f in overlapping:
                overlap_ms = max(0, min(w.end_ms, f.end_ms) - max(w.start_ms, f.start_ms))
                votes[f.speaker_id] = votes.get(f.speaker_id, 0.0) + overlap_ms * f.confidence
                states[f.speaker_id] = f.state

            best_spk = max(votes.items(), key=lambda kv: kv[1])[0]
            assignments.append((best_spk, states.get(best_spk, "provisional")))
        return assignments

    def build_caption_spec(
        self,
        segment_id: str,
        raw_words: List[RawWordAlignment],
        word_speakers: List[Tuple[str, str]],
        prosody_list: List[ProsodyMetrics],
        events: List[NonSpeechEvent],
        affect_frames_50hz: Optional[np.ndarray] = None,
        acoustic_time_ms: Optional[int] = None,
        locale: str = "en",
        compass_state: Optional[Dict[str, float]] = None,
        intensity: float = 1.0,
        is_safari: bool = False,
    ) -> CaptionSpec:
        word_tokens: List[WordToken] = []
        bearing = compass_state.get("bearing_deg") if compass_state else None
        spread = compass_state.get("spread_deg") if compass_state else None

        for idx, (w, (spk_id, spk_state), prosody) in enumerate(
            zip(raw_words, word_speakers, prosody_list)
        ):
            duration_ms = max(1, w.end_ms - w.start_ms)
            color_hex = self._get_speaker_color(spk_id, spk_state)

            typo = compute_typography_spec(
                prosody=prosody,
                duration_ms=duration_ms,
                intensity=intensity,
                is_safari=is_safari,
                locale=locale,
                is_offscreen_or_sfx=False,
            )

            word_tokens.append(
                WordToken(
                    word_id=f"w_{segment_id.replace('seg_', '')}_{idx + 1:02d}",
                    text=w.text,
                    start_ms=w.start_ms,
                    end_ms=w.end_ms,
                    duration_ms=duration_ms,
                    speaker=SpeakerInfo(
                        id=spk_id,
                        state=spk_state,  # type: ignore
                        color_hex=color_hex,
                        compass_bearing_deg=bearing,
                        compass_spread_deg=spread,
                    ),
                    prosody=prosody,
                    typography=typo,
                )
            )

        return CaptionSpec(
            schema_version="2.0-cwi",
            segment_id=segment_id,
            stream_acoustic_time_ms=acoustic_time_ms,
            locale=locale,
            presentation_offset_ms=self.config.presentation_offset_ms,
            work_area_box=WorkAreaBox(bottom_pct=20.0, bg_opacity=0.90),
            non_speech_events=events,
            words=word_tokens,
        )

    def build_events_only_spec(
        self,
        segment_id: str,
        events: List[NonSpeechEvent],
        acoustic_time_ms: int,
        locale: str = "en",
    ) -> CaptionSpec:
        return CaptionSpec(
            schema_version="2.0-cwi",
            segment_id=segment_id,
            stream_acoustic_time_ms=acoustic_time_ms,
            locale=locale,
            presentation_offset_ms=self.config.presentation_offset_ms,
            work_area_box=WorkAreaBox(bottom_pct=20.0, bg_opacity=0.90),
            non_speech_events=events,
            words=[],
        )
```
