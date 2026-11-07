"""
Pytest Fixtures for Caption with Intention (CWI) Verification Suite
References:
  - docs/00_overview/folders.md
  - docs/01_contracts_and_schemas/caption_spec_schema.md
  - docs/01_contracts_and_schemas/typography_and_math_constants.md
"""

import pytest
import numpy as np


@pytest.fixture
def sample_caption_spec_dict():
    """Canonical CaptionSpec v2.0-cwi reference dictionary from Section 1 of caption_spec_schema.md."""
    return {
        "schema_version": "2.0-cwi",
        "segment_id": "seg_0042",
        "presentation_offset_ms": 1750,
        "work_area_box": {
            "bottom_pct": 20.0,
            "bg_opacity": 0.90,
        },
        "non_speech_events": [
            {
                "type": "BGM",
                "label": "<|BGM|>",
                "start_ms": 12000,
                "end_ms": 18500,
                "confidence": 1.0,
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
                    "compass_bearing_deg": -35.0,
                    "compass_spread_deg": 12.0,
                },
                "prosody": {
                    "rms_db": -8.4,
                    "f0_word_hz": 215.0,
                    "f0_speaker_median_hz": 110.0,
                    "prominence_score": 0.94,
                    "isolated_silence_hold": True,
                },
                "typography": {
                    "crest_scale": 1.48,
                    "crest_sustain_scale": 1.036,
                    "pop_scale": 1.15,
                    "wght": 820,
                    "wdth": 128,
                    "grad": 85,
                    "hold_lift_em": -0.525,
                },
            }
        ],
    }


@pytest.fixture
def speaker_palette_data():
    """Official CWI high-contrast speaker color palette from Section 4 of typography_and_math_constants.md."""
    return [
        {"slot": 0, "id": "SPK_00", "name": "CWI Electric Cyan", "color_hex": "#4FC3F7", "ass_bgr": "&H00F7C34F&"},
        {"slot": 1, "id": "SPK_01", "name": "CWI Amber Gold",    "color_hex": "#FFD54F", "ass_bgr": "&H004FD5FF&"},
        {"slot": 2, "id": "SPK_02", "name": "CWI Coral Rose",    "color_hex": "#FF8A80", "ass_bgr": "&H00808AFF&"},
        {"slot": 3, "id": "SPK_03", "name": "CWI Mint Emerald",  "color_hex": "#69F0AE", "ass_bgr": "&H00AEF069&"},
        {"slot": 4, "id": "SPK_04", "name": "CWI Lavender Violet","color_hex": "#B388FF", "ass_bgr": "&H00FF88B3&"},
        {"slot": 5, "id": "SPK_05", "name": "CWI Tangerine",     "color_hex": "#FFAB40", "ass_bgr": "&H0040ABFF&"},
    ]


@pytest.fixture
def synthetic_audio_pcm_16k():
    """Generates 1.0 second of 16,000 Hz float32 mono audio (440 Hz test tone at -12 dBFS)."""
    sample_rate = 16000
    duration_sec = 1.0
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
    tone = 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
    return tone.astype(np.float32)
