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
