from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


SpeakerState = Literal["unknown", "provisional", "stable", "corrected"]

NonSpeechEventType = Literal[
    "BGM",
    "APPLAUSE",
    "LAUGHTER",
    "CRYING",
    "SIGH",
    "GASP",
    "COUGH",
    "SNEEZE",
    "BREATHING",
    "SCREAM",
    "YAWN",
    "THROAT_CLEARING",
    "SNIFF",
]

NonSpeechEventLabel = Literal[
    "<|BGM|>",
    "<|Applause|>",
    "[laughter]",
    "[crying]",
    "[sighs]",
    "[gasps]",
    "[coughing]",
    "[sneezing]",
    "[breathing]",
    "[screaming]",
    "[yawning]",
    "[throat clearing]",
    "[sniffs]",
]


class WorkAreaBox(BaseModel):
    bottom_pct: float = Field(default=20.0, ge=20.0, le=20.0)
    bg_opacity: float = Field(default=0.90, ge=0.90, le=0.90)


class NonSpeechEvent(BaseModel):
    type: NonSpeechEventType
    label: NonSpeechEventLabel
    start_ms: int = Field(..., ge=0)
    end_ms: int = Field(..., ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_span(self) -> "NonSpeechEvent":
        if self.end_ms < self.start_ms:
            raise ValueError(f"end_ms ({self.end_ms}) must be >= start_ms ({self.start_ms})")
        return self


class SpeakerInfo(BaseModel):
    id: str = Field(..., min_length=1)
    state: SpeakerState = Field(default="unknown")
    color_hex: str = Field(..., pattern=r"^#[0-9A-Fa-f]{6}$")
    compass_bearing_deg: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    compass_spread_deg: Optional[float] = Field(default=None, ge=0.0, le=180.0)


class ProsodyMetrics(BaseModel):
    rms_db: float = Field(..., ge=-100.0, le=0.0)
    f0_word_hz: float = Field(..., ge=0.0, le=1000.0)
    f0_speaker_median_hz: float = Field(..., ge=50.0, le=500.0)
    prominence_score: float = Field(..., ge=0.0, le=1.0)
    isolated_silence_hold: bool = Field(default=False)
    pitch_contour_norm: Optional[List[float]] = Field(default=None)


class AudioLoudnessStats(BaseModel):
    floor_db: float = Field(default=-42.0, ge=-100.0, le=0.0)
    conversational_median_db: float = Field(default=-20.0, ge=-100.0, le=0.0)
    ceiling_db: float = Field(default=-4.0, ge=-100.0, le=0.0)
    deadband_db: float = Field(default=2.0, ge=0.0, le=12.0)
    p15_rms_db: Optional[float] = None
    p85_rms_db: Optional[float] = None
    p95_rms_db: Optional[float] = None

    def __init__(
        self,
        floor_db: Optional[float] = None,
        conversational_median_db: Optional[float] = None,
        ceiling_db: Optional[float] = None,
        median_speech_rms_db: Optional[float] = None,
        whisper_rms_db: Optional[float] = None,
        shout_rms_db: Optional[float] = None,
        deadband_db: float = 2.0,
        p15_rms_db: Optional[float] = None,
        p85_rms_db: Optional[float] = None,
        p95_rms_db: Optional[float] = None,
        **data,
    ):
        med = conversational_median_db if conversational_median_db is not None else (median_speech_rms_db if median_speech_rms_db is not None else -20.0)
        flr = floor_db if floor_db is not None else (whisper_rms_db if whisper_rms_db is not None else -42.0)
        ceil = ceiling_db if ceiling_db is not None else (shout_rms_db if shout_rms_db is not None else -4.0)
        super().__init__(
            floor_db=flr,
            conversational_median_db=med,
            ceiling_db=ceil,
            deadband_db=deadband_db,
            p15_rms_db=p15_rms_db,
            p85_rms_db=p85_rms_db,
            p95_rms_db=p95_rms_db,
            **data,
        )

    @property
    def median_speech_rms_db(self) -> float:
        return self.conversational_median_db

    @property
    def whisper_rms_db(self) -> float:
        return self.floor_db

    @property
    def shout_rms_db(self) -> float:
        return self.ceiling_db

    @classmethod
    def from_rms_array(cls, rms_db_list: List[float]) -> "AudioLoudnessStats":
        import numpy as np
        valid = [r for r in rms_db_list if r > -55.0]
        if not valid:
            return cls()
        arr = np.array(valid, dtype=np.float32)
        floor = float(np.percentile(arr, 15))
        median = float(np.median(arr))
        ceil = float(np.percentile(arr, 95))
        p85 = float(np.percentile(arr, 85))
        whisper = min(floor, median - 6.0)
        shout = max(ceil, median + 8.0)
        return cls(
            floor_db=round(whisper, 2),
            conversational_median_db=round(median, 2),
            ceiling_db=round(shout, 2),
            deadband_db=2.0,
            p15_rms_db=round(floor, 2),
            p85_rms_db=round(p85, 2),
            p95_rms_db=round(ceil, 2),
        )


class TypographySpec(BaseModel):
    crest_scale: float = Field(..., ge=0.72, le=1.62)
    crest_sustain_scale: float = Field(..., ge=0.50, le=1.62)
    pop_scale: float = Field(default=1.15, ge=1.15, le=1.15)
    wght: int = Field(..., ge=340, le=900)
    wdth: int = Field(..., ge=25, le=151)
    grad: int = Field(default=0, ge=-200, le=150)
    hold_lift_em: float = Field(default=0.0, ge=-0.525, le=0.0)
    slnt: float = Field(default=0.0, ge=-10.0, le=0.0)
    opsz: float = Field(default=14.0, ge=8.0, le=144.0)
    xtra: int = Field(default=468, ge=323, le=603)
    xopq: int = Field(default=96, ge=27, le=175)
    yopq: int = Field(default=79, ge=25, le=135)
    ytas: int = Field(default=750, ge=649, le=854)
    ytde: int = Field(default=-203, ge=-305, le=-98)


class WordToken(BaseModel):
    word_id: str = Field(..., pattern=r"^w_[a-zA-Z0-9_]+$")
    text: str = Field(..., min_length=1)
    is_disfluency: bool = Field(default=False)
    start_ms: int = Field(..., ge=0)
    end_ms: int = Field(..., ge=1)
    duration_ms: int = Field(..., ge=1)
    speaker: SpeakerInfo
    prosody: ProsodyMetrics
    typography: TypographySpec

    @model_validator(mode="after")
    def validate_timing(self) -> "WordToken":
        expected_dur = self.end_ms - self.start_ms
        if expected_dur <= 0:
            raise ValueError(f"end_ms ({self.end_ms}) must be strictly > start_ms ({self.start_ms})")
        if self.duration_ms != expected_dur:
            self.duration_ms = expected_dur
        return self


class CaptionSpec(BaseModel):
    schema_version: Literal["2.0-cwi"] = "2.0-cwi"
    segment_id: str = Field(..., pattern=r"^seg_[a-zA-Z0-9_]+$")
    stream_acoustic_time_ms: Optional[int] = Field(default=None, ge=0)
    locale: str = Field(default="en")
    presentation_offset_ms: int = Field(default=1750, ge=420, le=5000)
    work_area_box: WorkAreaBox = Field(default_factory=WorkAreaBox)
    non_speech_events: List[NonSpeechEvent] = Field(default_factory=list)
    words: List[WordToken] = Field(default_factory=list)
    loudness_stats: Optional[AudioLoudnessStats] = None
