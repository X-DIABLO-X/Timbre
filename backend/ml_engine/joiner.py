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
                    is_disfluency=w.is_disfluency,
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
