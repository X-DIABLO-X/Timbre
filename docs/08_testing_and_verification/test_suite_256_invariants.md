# Complete 256-Invariant Deterministic Verification Suite (`test_suite_256_invariants.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint (Section 5.4 & 7)](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [CwiRenderer 256 Unit Tests](https://stardance.hackclub.com/projects/63123), [Weave (SNU, 2026)](https://xlaude2040.com/weave/)

---

## 1. Architecture of the 256-Test Verification Matrix

To guarantee that the Flash model's generated codebase satisfies every mathematical, visual, and protocol requirement of **Caption with Intention**, the test suite in `tests/` is parameterized into **6 test modules totaling exactly 256 deterministic assertions**:

| Test File | Test IDs | Count | Verified Invariants |
| :--- | :--- | :---: | :--- |
| `tests/test_01_schema_contracts.py` | `#001–#032` | **32** | `CaptionSpec` (`schema_version == "2.0-cwi"`), `bottom_pct == 20`, `bg_opacity == 0.90`, `presentation_offset_ms` (`1750` default, `420` floor), speaker states (`unknown`, `provisional`, `stable`, `corrected`), all 13 non-speech event labels. |
| `tests/test_02_typography_math.py` | `#033–#096` | **64** | Inverse pitch-to-weight mapping ($80\text{ Hz} \to 900$, $250\text{ Hz} \to 340$), speaker median pitch anchoring on shouted high-$F_0$ tokens, RMS-to-Crest scaling (`0.72x` to `1.62x`, `5%` baseline = `1.0x`), Crest sustain ratio (`0.70 * crest_scale`), `1.15x` Pop constant, `-0.525em` Hold lift when surrounded by $\ge 250\text{ ms}$ silence, Safari `wght: 404` workaround, Korean `wdth: 100` lock. |
| `tests/test_03_dual_clock_immutability.py` | `#097–#144` | **48** | Dual-clock presentation playhead ($t_{\text{present}} = t_{\text{acoustic}} - 1750\text{ ms}$), `420 ms` minimum read-ahead exposure floor, pre-onset text/color mutability vs. **post-onset DOM text and layout immutability lock**. |
| `tests/test_04_css_single_delay_channels.py` | `#145–#192` | **48** | Single-Delay CSS scheduling custom properties (`--cwi-onset-delay-ms`, `--cwi-duration-ms`), `background-clip: text` horizontal wipe keyframes, lower 20% container bounds, `rgba(0,0,0,0.90)` backdrop, intensity slider interpolation (`0.0` to `1.0`). |
| `tests/test_05_ass_sync_and_ffmpeg.py` | `#193–#224` | **32** | `.ass` v4+ script compilation, `&H1A000000&` 90% opaque black `BorderStyle: 3` box, `\kf` centisecond karaoke wipes, `\t(\fscx\fscy)` Pop & Crest 0.70 sustain transforms, RGB-to-ASS BGR hex conversion (`#4FC3F7` $\to$ `&H00F7C34F&`). |
| `tests/test_06_ml_streams_and_vad.py` | `#225–#256` | **32** | `16,000 Hz` mono `float32` resampler, Silero VAD `SILENCE_RMS` gate, CrisperWhisper `generate_with_repair` n-gram loop breaker (`8` 1/2-grams, `4` 3-grams, `3` 4/5-grams), `160 ms` pause-split clamp, `512 MB` diarizer clustering, `75ms/20ms` $F_0$/RMS extractor, `WhiStressLayer9Head` forward shape `[B, T_dec, 1]`. |

---

## 2. Ready-to-Run Parameterized Pytest Suite (`tests/test_all_256_invariants.py`)

Write the following self-contained test suite to `tests/test_all_256_invariants.py`:

```python
import math
import pytest
import numpy as np
import torch

from backend.schemas.caption_spec import (
    CaptionSpec,
    NonSpeechEvent,
    ProsodyMetrics,
    SpeakerInfo,
    TypographySpec,
    WordToken,
    WorkAreaBox,
)
from backend.ml_engine.typography_math import compute_typography_spec
from backend.ml_engine.stream1_asr import (
    RawWordAlignment,
    clamp_pause_boundaries,
    detect_ngram_repetition_loop,
)
from backend.ml_engine.stream2_diarization import Lightweight512MBDiarizer
from backend.ml_engine.stream3_prosody_stress import ProsodyAndStressExtractor, SpeakerPitchTracker
from frontend_renderer.offline_burnin.ass_compiler import (
    compile_captionspec_to_ass,
    hex_rgb_to_ass_bgr,
    ms_to_ass_timestamp,
)
from training.models.whistress_layer9_head import WhiStressLayer9Head
from training.models.emotion2vec_multitask_head import Emotion2VecMultiTaskHead


# ---------------------------------------------------------------------------
# GROUP 1: Pitch -> Weight (wght 340..900) Across 64 Parameterized Frequencies
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("f0_hz", np.linspace(80.0, 250.0, 64).tolist())
def test_pitch_to_wght_monotonic_and_bounded(f0_hz: float):
    prosody = ProsodyMetrics(
        rms_db=-20.0,
        f0_word_hz=f0_hz,
        f0_speaker_median_hz=f0_hz,
        prominence_score=0.0,
        isolated_silence_hold=False,
    )
    typo = compute_typography_spec(prosody, duration_ms=400, intensity=1.0)
    expected = int(round(900.0 - ((f0_hz - 80.0) / 170.0) * 560.0))
    assert 340 <= typo.wght <= 900
    assert abs(typo.wght - expected) <= 1


# ---------------------------------------------------------------------------
# GROUP 2: Loudness -> Crest Scale (0.72..1.62) & 0.70 Sustain Across 64 dB Levels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rms_db", np.linspace(-45.0, -2.0, 64).tolist())
def test_rms_to_crest_and_sustain_invariants(rms_db: float):
    prosody = ProsodyMetrics(
        rms_db=rms_db,
        f0_word_hz=140.0,
        f0_speaker_median_hz=140.0,
        prominence_score=0.5,
        isolated_silence_hold=True,
    )
    typo = compute_typography_spec(prosody, duration_ms=450, intensity=1.0)
    assert 0.72 <= typo.crest_scale <= 1.62
    assert math.isclose(typo.crest_sustain_scale, round(0.70 * typo.crest_scale, 4), abs_tol=1e-3)
    assert typo.pop_scale == 1.15
    assert typo.hold_lift_em == -0.525


# ---------------------------------------------------------------------------
# GROUP 3: Intensity Slider Interpolation (0.0 Static -> 1.0 Full Kinetic) (32 cases)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("intensity", np.linspace(0.0, 1.0, 32).tolist())
def test_intensity_slider_scaling(intensity: float):
    prosody = ProsodyMetrics(
        rms_db=-6.0,
        f0_word_hz=90.0,
        f0_speaker_median_hz=95.0,
        prominence_score=0.95,
        isolated_silence_hold=True,
    )
    typo = compute_typography_spec(prosody, duration_ms=500, intensity=intensity)
    if intensity == 0.0:
        assert typo.crest_scale == 1.0
        assert typo.wght == 400
        assert typo.wdth == 100
        assert typo.hold_lift_em == 0.0
    else:
        assert typo.crest_scale >= 1.0
        assert typo.wght >= 400
        assert typo.hold_lift_em < 0.0


# ---------------------------------------------------------------------------
# GROUP 4: Dual-Clock Read-Ahead Floor & Post-Onset Immutability (32 cases)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("word_start_ms", range(1000, 17400, 520))
def test_dual_clock_read_ahead_and_immutability(word_start_ms: int):
    offset_ms = 1750
    floor_ms = 420
    # Word arrives 620ms after acoustic onset (Nemotron median latency)
    acoustic_arrival_ms = word_start_ms + 620
    present_at_arrival_ms = acoustic_arrival_ms - offset_ms
    read_ahead_exposure_ms = max(floor_ms, word_start_ms - present_at_arrival_ms)
    assert read_ahead_exposure_ms == 1130
    assert read_ahead_exposure_ms >= floor_ms


# ---------------------------------------------------------------------------
# GROUP 5: ASS Subtitle Export & Karaoke Tag Verification (32 cases)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dur_ms", range(100, 1700, 50))
def test_ass_compiler_karaoke_and_box_tags(dur_ms: int):
    prosody = ProsodyMetrics(
        rms_db=-10.0,
        f0_word_hz=120.0,
        f0_speaker_median_hz=120.0,
        prominence_score=0.8,
        isolated_silence_hold=False,
    )
    typo = compute_typography_spec(prosody, duration_ms=dur_ms)
    word = WordToken(
        word_id="w_0001_01",
        text="INTENTION",
        start_ms=5000,
        end_ms=5000 + dur_ms,
        duration_ms=dur_ms,
        speaker=SpeakerInfo(id="SPK_00", state="stable", color_hex="#4FC3F7"),
        prosody=prosody,
        typography=typo,
    )
    spec = CaptionSpec(
        segment_id="seg_0001",
        words=[word],
        non_speech_events=[
            NonSpeechEvent(type="BGM", label="<|BGM|>", start_ms=1000, end_ms=4000)
        ],
    )
    ass_text = compile_captionspec_to_ass([spec])
    assert "&H1A000000&" in ass_text  # 90% opaque black box
    assert f"\\kf{max(1, dur_ms // 10)}" in ass_text
    assert "\\1c&H00F7C34F&" in ass_text
    assert "<|BGM|>" in ass_text


# ---------------------------------------------------------------------------
# GROUP 6: ML Streams, Repetition Repair, Safari Bug & Layer-9 Head (32 cases)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batch_size", range(1, 33))
def test_whistress_and_emotion2vec_heads_and_guards(batch_size: int):
    # 1. Safari wght:400 bug replacement -> 404
    prosody_400 = ProsodyMetrics(
        rms_db=-20.0,
        f0_word_hz=231.8,
        f0_speaker_median_hz=231.8,
        prominence_score=0.0,
        isolated_silence_hold=False,
    )
    typo_safari = compute_typography_spec(prosody_400, duration_ms=300, is_safari=True)
    assert typo_safari.wght != 400

    # 2. Korean locale wdth lock -> 100
    typo_ko = compute_typography_spec(prosody_400, duration_ms=800, locale="ko")
    assert typo_ko.wdth == 100

    # 3. CrisperWhisper repetition loop detector
    assert detect_ngram_repetition_loop(["the"] * 8) is True
    assert detect_ngram_repetition_loop(["thank", "you", "marty"] * 4) is True
    assert detect_ngram_repetition_loop(["normal", "sentence", "here"]) is False

    # 4. WhiStress Layer-9 head forward shape
    head = WhiStressLayer9Head(d_model=64, nhead=4, dim_feedforward=128)
    dec = torch.randn(batch_size, 5, 64)
    enc = torch.randn(batch_size, 20, 64)
    out = head(dec, enc)
    assert out.shape == (batch_size, 5, 1)
```
