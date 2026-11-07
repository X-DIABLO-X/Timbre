# `CaptionSpec v2.0-cwi` Data Contract, Schemas & Wire Protocol (`caption_spec_schema.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint (Section 5.2)](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [Weave (SNU, 2026)](https://xlaude2040.com/weave/), [CwiRenderer](https://stardance.hackclub.com/projects/63123)

---

## 1. Overview of the `CaptionSpec` Contract

The `CaptionSpec` (`schema_version: "2.0-cwi"`) JSON payload is the **single source of truth** coupling Part 1 (The Parallel ML Audio Engine) and Part 2 (The Deterministic `CwiRenderer` & Offline `.ass` Sync Engine). Every WebSocket message, Server-Sent Event (SSE), WebGPU Worker message, and offline `.cwi.json` sidecar file must conform strictly to this schema.

### Canonical Reference Payload (`CaptionSpec v2.0-cwi`)

```json
{
  "schema_version": "2.0-cwi",
  "segment_id": "seg_0042",
  "presentation_offset_ms": 1750,
  "work_area_box": {
    "bottom_pct": 20,
    "bg_opacity": 0.90
  },
  "non_speech_events": [
    {
      "type": "BGM",
      "label": "<|BGM|>",
      "start_ms": 12000,
      "end_ms": 18500
    }
  ],
  "words": [
    {
      "word_id": "w_0042_01",
      "text": "THANK",
      "start_ms": 14200,
      "end_ms": 14680,
      "duration_ms": 480,
      "speaker": {
        "id": "SPK_01",
        "state": "stable",
        "color_hex": "#4285F4",
        "compass_bearing_deg": -35,
        "compass_spread_deg": 12
      },
      "prosody": {
        "rms_db": -8.4,
        "f0_word_hz": 215.0,
        "f0_speaker_median_hz": 110.0,
        "prominence_score": 0.94,
        "isolated_silence_hold": true
      },
      "typography": {
        "crest_scale": 1.48,
        "crest_sustain_scale": 1.036,
        "pop_scale": 1.15,
        "wght": 820,
        "wdth": 128,
        "grad": 85,
        "hold_lift_em": -0.525
      }
    }
  ]
}
```

---

## 2. Complete JSON Schema (Draft 2020-12) — `shared_schemas/caption_spec.schema.json`

Write the following exact JSON Schema to `shared_schemas/caption_spec.schema.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://captionwithintention.org/schemas/v2.0-cwi/caption_spec.schema.json",
  "title": "CaptionSpec",
  "description": "Versioned JSON contract connecting the Parallel ML Audio Engine and the Deterministic CwiRenderer.",
  "type": "object",
  "required": [
    "schema_version",
    "segment_id",
    "presentation_offset_ms",
    "work_area_box",
    "non_speech_events",
    "words"
  ],
  "additionalProperties": false,
  "properties": {
    "schema_version": {
      "type": "string",
      "const": "2.0-cwi"
    },
    "segment_id": {
      "type": "string",
      "pattern": "^seg_[a-zA-Z0-9_]+$"
    },
    "stream_acoustic_time_ms": {
      "type": "integer",
      "minimum": 0,
      "description": "Optional current acoustic clock timestamp in ms when emitted in live streaming mode."
    },
    "locale": {
      "type": "string",
      "default": "en",
      "description": "BCP-47 language code (e.g., 'en', 'ko'). Controls Roboto Flex vs. Noto Sans KR font selection."
    },
    "presentation_offset_ms": {
      "type": "integer",
      "minimum": 420,
      "maximum": 5000,
      "default": 1750,
      "description": "Dual-clock read-ahead offset in milliseconds (default 1750ms, minimum floor 420ms)."
    },
    "work_area_box": {
      "type": "object",
      "required": ["bottom_pct", "bg_opacity"],
      "additionalProperties": false,
      "properties": {
        "bottom_pct": {
          "type": "number",
          "const": 20,
          "description": "Strictly restricted to the lower 20% work area of the video viewport."
        },
        "bg_opacity": {
          "type": "number",
          "const": 0.90,
          "description": "90% opacity black caption box (rgba(0, 0, 0, 0.90))."
        }
      }
    },
    "non_speech_events": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["type", "label", "start_ms", "end_ms"],
        "additionalProperties": false,
        "properties": {
          "type": {
            "type": "string",
            "enum": [
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
              "SNIFF"
            ]
          },
          "label": {
            "type": "string",
            "enum": [
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
              "[sniffs]"
            ]
          },
          "start_ms": {
            "type": "integer",
            "minimum": 0
          },
          "end_ms": {
            "type": "integer",
            "minimum": 0
          },
          "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 1.0
          }
        }
      }
    },
    "words": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "word_id",
          "text",
          "start_ms",
          "end_ms",
          "duration_ms",
          "speaker",
          "prosody",
          "typography"
        ],
        "additionalProperties": false,
        "properties": {
          "word_id": {
            "type": "string",
            "pattern": "^w_[a-zA-Z0-9_]+$"
          },
          "text": {
            "type": "string",
            "minLength": 1
          },
          " is_disfluency": {
            "type": "boolean",
            "default": false
          },
          "start_ms": {
            "type": "integer",
            "minimum": 0
          },
          "end_ms": {
            "type": "integer",
            "minimum": 1
          },
          "duration_ms": {
            "type": "integer",
            "minimum": 1
          },
          "speaker": {
            "type": "object",
            "required": ["id", "state", "color_hex"],
            "additionalProperties": false,
            "properties": {
              "id": {
                "type": "string",
                "description": "Speaker identifier (e.g., 'SPK_00', 'SPK_01', 'SPK_UNKNOWN')."
              },
              "state": {
                "type": "string",
                "enum": ["unknown", "provisional", "stable", "corrected"],
                "description": "Weave 4-state attribution lifecycle."
              },
              "color_hex": {
                "type": "string",
                "pattern": "^#[0-9A-Fa-f]{6}$",
                "description": "Assigned speaker hex color (or #9AA0A6 neutral grey when state is 'unknown')."
              },
              "compass_bearing_deg": {
                "type": ["number", "null"],
                "minimum": -180.0,
                "maximum": 180.0,
                "default": null,
                "description": "Azimuth angle in degrees from wearable mic array (-180 to +180, 0 is straight ahead)."
              },
              "compass_spread_deg": {
                "type": ["number", "null"],
                "minimum": 0.0,
                "maximum": 180.0,
                "default": null,
                "description": "Angular spread of microphone DoA observations in degrees."
              }
            }
          },
          "prosody": {
            "type": "object",
            "required": [
              "rms_db",
              "f0_word_hz",
              "f0_speaker_median_hz",
              "prominence_score",
              "isolated_silence_hold"
            ],
            "additionalProperties": false,
            "properties": {
              "rms_db": {
                "type": "number",
                "minimum": -100.0,
                "maximum": 0.0
              },
              "f0_word_hz": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1000.0
              },
              "f0_speaker_median_hz": {
                "type": "number",
                "minimum": 50.0,
                "maximum": 500.0
              },
              "prominence_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
              },
              "isolated_silence_hold": {
                "type": "boolean"
              },
              "pitch_contour_norm": {
                "type": "array",
                "items": {
                  "type": "number",
                  "minimum": 0.0,
                  "maximum": 1.0
                },
                "description": "Optional normalized intra-word pitch contour samples driving Roboto Flex wdth modulation."
              }
            }
          },
          "typography": {
            "type": "object",
            "required": [
              "crest_scale",
              "crest_sustain_scale",
              "pop_scale",
              "wght",
              "wdth",
              "grad",
              "hold_lift_em"
            ],
            "additionalProperties": false,
            "properties": {
              "crest_scale": {
                "type": "number",
                "minimum": 0.72,
                "maximum": 1.62
              },
              "crest_sustain_scale": {
                "type": "number",
                "minimum": 0.50,
                "maximum": 1.62
              },
              "pop_scale": {
                "type": "number",
                "const": 1.15
              },
              "wght": {
                "type": "integer",
                "minimum": 340,
                "maximum": 900
              },
              "wdth": {
                "type": "integer",
                "minimum": 25,
                "maximum": 151
              },
              "grad": {
                "type": "integer",
                "minimum": -200,
                "maximum": 150
              },
              "hold_lift_em": {
                "type": "number",
                "minimum": -0.525,
                "maximum": 0.0
              },
              "slnt": {
                "type": "number",
                "minimum": -10.0,
                "maximum": 0.0,
                "default": 0.0
              },
              "opsz": {
                "type": "number",
                "minimum": 8.0,
                "maximum": 144.0,
                "default": 14.0
              },
              "xtra": {
                "type": "integer",
                "minimum": 323,
                "maximum": 603,
                "default": 468
              },
              "xopq": {
                "type": "integer",
                "minimum": 27,
                "maximum": 175,
                "default": 96
              },
              "yopq": {
                "type": "integer",
                "minimum": 25,
                "maximum": 135,
                "default": 79
              },
              "ytas": {
                "type": "integer",
                "minimum": 649,
                "maximum": 854,
                "default": 750
              },
              "ytde": {
                "type": "integer",
                "minimum": -305,
                "maximum": -98,
                "default": -203
              }
            }
          }
        }
      }
    }
  }
}
```

---

## 3. Python Pydantic v2 Implementation — `backend/schemas/caption_spec.py`

```python
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
```

---

## 4. TypeScript Interfaces — `frontend_renderer/src/core/types.ts`

```typescript
export type SpeakerState = 'unknown' | 'provisional' | 'stable' | 'corrected';

export type NonSpeechEventType =
  | 'BGM'
  | 'APPLAUSE'
  | 'LAUGHTER'
  | 'CRYING'
  | 'SIGH'
  | 'GASP'
  | 'COUGH'
  | 'SNEEZE'
  | 'BREATHING'
  | 'SCREAM'
  | 'YAWN'
  | 'THROAT_CLEARING'
  | 'SNIFF';

export type NonSpeechEventLabel =
  | '<|BGM|>'
  | '<|Applause|>'
  | '[laughter]'
  | '[crying]'
  | '[sighs]'
  | '[gasps]'
  | '[coughing]'
  | '[sneezing]'
  | '[breathing]'
  | '[screaming]'
  | '[yawning]'
  | '[throat clearing]'
  | '[sniffs]';

export interface WorkAreaBox {
  readonly bottom_pct: 20;
  readonly bg_opacity: 0.90;
}

export interface NonSpeechEvent {
  type: NonSpeechEventType;
  label: NonSpeechEventLabel;
  start_ms: number;
  end_ms: number;
  confidence?: number;
}

export interface SpeakerInfo {
  id: string;
  state: SpeakerState;
  color_hex: string;
  compass_bearing_deg?: number | null;
  compass_spread_deg?: number | null;
}

export interface ProsodyMetrics {
  rms_db: number;
  f0_word_hz: number;
  f0_speaker_median_hz: number;
  prominence_score: number;
  isolated_silence_hold: boolean;
  pitch_contour_norm?: number[];
}

export interface TypographySpec {
  crest_scale: number;         // [0.72, 1.62]
  crest_sustain_scale: number; // 0.70 * crest_scale (clamped [0.50, 1.62])
  pop_scale: 1.15;             // Constant 1.15x onset impulse
  wght: number;                // [340, 900]
  wdth: number;                // [25, 151]
  grad: number;                // [-200, 150]
  hold_lift_em: number;        // -0.525 if isolated_silence_hold else 0.0
  slnt?: number;               // [-10, 0]
  opsz?: number;               // [8, 144]
  xtra?: number;               // [323, 603]
  xopq?: number;               // [27, 175]
  yopq?: number;               // [25, 135]
  ytas?: number;               // [649, 854]
  ytde?: number;               // [-305, -98]
}

export interface WordToken {
  word_id: string;
  text: string;
  start_ms: number;
  end_ms: number;
  duration_ms: number;
  speaker: SpeakerInfo;
  prosody: ProsodyMetrics;
  typography: TypographySpec;
}

export interface CaptionSpec {
  schema_version: '2.0-cwi';
  segment_id: string;
  stream_acoustic_time_ms?: number;
  locale?: string;
  presentation_offset_ms: number;
  work_area_box: WorkAreaBox;
  non_speech_events: NonSpeechEvent[];
  words: WordToken[];
}
```

---

## 5. WebSocket Wire Framing Protocol (`ws://127.0.0.1:8765/ws`)

The WebSocket connection between `extension/offscreen/offscreen.js` (or the React VOD player) and `backend/main.py` multiplexes binary PCM audio frames upstream and JSON `CaptionSpec` messages downstream:

1. **Client $\to$ Server (Control Frame — JSON Text):**
   ```json
   {
     "type": "SESSION_START",
     "session_id": "tab_4910",
     "sample_rate": 16000,
     "channels": 1,
     "format": "pcm_s16le",
     "mode": "live_streaming",
     "locale": "en",
     "enable_stem_separation": false
   }
   ```
2. **Client $\to$ Server (Audio Chunk Frame — Binary ArrayBuffer):**
   - Headerless raw `16,000 Hz`, mono, `Int16` (`pcm_s16le`) or `Float32` (`pcm_f32le`) audio frames emitted every `100 ms` (`1,600` samples = `3,200 bytes` in `Int16` or `6,400 bytes` in `Float32`, matching [Pyannote Streaming](https://www.pyannote.ai/blog/how-we-built-streaming-diarization)'s exact `6,400-byte` chunk size).
3. **Server $\to$ Client (CaptionSpec Payload — JSON Text):**
   - Full valid `CaptionSpec` (`schema_version: "2.0-cwi"`) JSON object containing newly emitted read-ahead words, updated speaker attribution states (`unknown` $\to$ `provisional` $\to$ `stable` $\to$ `corrected`), and non-speech events.
