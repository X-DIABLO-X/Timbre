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
