"""
Deterministic Frame/Timestamp Quantizer & Sync Engine for Offline VOD
References:
  - docs/00_overview/folders.md
  - docs/01_contracts_and_schemas/typography_and_math_constants.md
  - docs/07_client_and_rendering/ffmpeg_ass_burnin.md
"""

from typing import List, Optional

try:
    from backend.schemas.caption_spec import CaptionSpec, WordToken, NonSpeechEvent
except ImportError:
    from typing import Any
    CaptionSpec = Any  # type: ignore
    WordToken = Any    # type: ignore
    NonSpeechEvent = Any  # type: ignore


class DeterministicSyncEngine:
    """
    Deterministic frame/timestamp quantizer for offline VOD rendering:
    - Snaps timestamps to exact video frame boundaries (e.g., 23.976, 24, 25, 29.97, 30, 60 fps).
    - Enforces monotonically increasing non-overlapping word boundaries.
    - Clamps minimum word durations to at least one video frame duration.
    - Detects >= 250ms silence gaps before and after words to maintain isolated_silence_hold.
    """

    def __init__(
        self,
        fps: float = 30.0,
        presentation_offset_ms: int = 1750,
        min_read_ahead_floor_ms: int = 420,
    ):
        self.fps = fps
        self.frame_duration_ms = 1000.0 / fps
        self.presentation_offset_ms = presentation_offset_ms
        self.min_read_ahead_floor_ms = min_read_ahead_floor_ms

    def quantize_ms(self, ms: int) -> int:
        """Snaps a millisecond timestamp to the nearest video frame boundary."""
        if ms <= 0:
            return 0
        frame_idx = round(ms / self.frame_duration_ms)
        return int(round(frame_idx * self.frame_duration_ms))

    def quantize_word(self, word: WordToken, prev_end_ms: int = 0) -> WordToken:
        """Quantizes word start and end times, ensuring valid duration and monotonic ordering."""
        q_start = self.quantize_ms(word.start_ms)
        if q_start < prev_end_ms:
            q_start = prev_end_ms

        q_end = self.quantize_ms(word.end_ms)
        min_dur = max(1, int(round(self.frame_duration_ms)))
        if q_end <= q_start:
            q_end = q_start + min_dur

        word.start_ms = q_start
        word.end_ms = q_end
        word.duration_ms = q_end - q_start
        return word

    def quantize_caption_spec(self, spec: CaptionSpec) -> CaptionSpec:
        """Quantizes all words and non-speech events in a CaptionSpec."""
        prev_end = 0
        for i, word in enumerate(spec.words):
            self.quantize_word(word, prev_end)
            prev_end = word.end_ms

        # Re-evaluate isolated_silence_hold invariant (>= 250ms silence on both sides)
        n_words = len(spec.words)
        for i, word in enumerate(spec.words):
            pre_gap = word.start_ms - (spec.words[i - 1].end_ms if i > 0 else 0)
            post_gap = (spec.words[i + 1].start_ms - word.end_ms) if i < n_words - 1 else 1000
            is_isolated = pre_gap >= 250 and post_gap >= 250

            word.prosody.isolated_silence_hold = is_isolated
            if is_isolated:
                word.typography.hold_lift_em = -0.525
            else:
                word.typography.hold_lift_em = 0.0

        for ev in spec.non_speech_events:
            ev.start_ms = self.quantize_ms(ev.start_ms)
            ev.end_ms = max(ev.start_ms + int(round(self.frame_duration_ms)), self.quantize_ms(ev.end_ms))

        return spec

    def sync_vod_stream(self, specs: List[CaptionSpec]) -> List[CaptionSpec]:
        """Processes an ordered sequence of CaptionSpecs for offline export."""
        synced: List[CaptionSpec] = []
        for spec in specs:
            synced.append(self.quantize_caption_spec(spec))
        return synced
