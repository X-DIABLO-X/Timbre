"""
Comprehensive Verification Suite for Per-Video Adaptive Acoustic Normalization
Validates that:
  1. Conversational speech median anchors to 1.00x (normal subtitle font size) regardless of overall video mix loudness.
  2. Subtitles never universally shrink to tiny text on quieter mixes (such as Dune at -33 dBFS).
  3. Genuine whispers scale down to the readable floor (0.72x).
  4. Shouts / high-energy peaks scale up to 1.35x - 1.62x with ascender/descender compression.
  5. Loudness statistics computation accurately reflects dialogue distributions.
"""

import math
import numpy as np
import pytest

from backend.schemas.caption_spec import AudioLoudnessStats, ProsodyMetrics
from backend.ml_engine.typography_math import (
    compute_audio_loudness_stats,
    compute_typography_spec,
    DEFAULT_LOUDNESS_STATS,
)
from backend.ml_engine.stream3_prosody_stress import SpeechLoudnessTracker


def test_dune_conversational_speech_anchors_to_unity():
    """
    On a movie like Dune with dialogue mixed at -33 dBFS:
    Conversational speech must anchor to crest_scale == 1.00x,
    resolving the user's issue where subtitles became small and whisper-like.
    """
    dune_stats = AudioLoudnessStats(
        median_speech_rms_db=-33.0,
        whisper_rms_db=-42.0,
        shout_rms_db=-16.0,
        deadband_db=2.0,
    )

    # Conversational speech at median (-33 dBFS, normal prominence 0.0)
    prosody_normal = ProsodyMetrics(
        rms_db=-33.0,
        f0_word_hz=120.0,
        f0_speaker_median_hz=120.0,
        prominence_score=0.0,
        isolated_silence_hold=False,
    )
    typo_normal = compute_typography_spec(
        prosody_normal, duration_ms=400, intensity=1.0, loudness_stats=dune_stats
    )
    assert typo_normal.crest_scale == 1.00, f"Expected 1.00 for normal dialogue, got {typo_normal.crest_scale}"

    # Conversational speech within deadband (-32.0 dBFS)
    prosody_deadband = ProsodyMetrics(
        rms_db=-32.0,
        f0_word_hz=120.0,
        f0_speaker_median_hz=120.0,
        prominence_score=0.1,
        isolated_silence_hold=False,
    )
    typo_deadband = compute_typography_spec(
        prosody_deadband, duration_ms=400, intensity=1.0, loudness_stats=dune_stats
    )
    assert 0.98 <= typo_deadband.crest_scale <= 1.03


def test_genuine_whisper_scales_down_to_floor():
    """
    Only genuine whispers below conversational baseline scale down to 0.72x floor.
    """
    dune_stats = AudioLoudnessStats(
        median_speech_rms_db=-33.0,
        whisper_rms_db=-43.0,
        shout_rms_db=-16.0,
        deadband_db=2.0,
    )

    # Quiet whisper at -43 dBFS
    prosody_whisper = ProsodyMetrics(
        rms_db=-43.0,
        f0_word_hz=110.0,
        f0_speaker_median_hz=120.0,
        prominence_score=0.0,
        isolated_silence_hold=False,
    )
    typo_whisper = compute_typography_spec(
        prosody_whisper, duration_ms=400, intensity=1.0, loudness_stats=dune_stats
    )
    assert typo_whisper.crest_scale == 0.72
    assert typo_whisper.crest_sustain_scale == round(0.70 * 0.72, 4)

    # Sub-whisper at -50 dBFS clamped to 0.72 floor
    prosody_sub = ProsodyMetrics(
        rms_db=-50.0,
        f0_word_hz=100.0,
        f0_speaker_median_hz=120.0,
        prominence_score=0.0,
        isolated_silence_hold=False,
    )
    typo_sub = compute_typography_spec(
        prosody_sub, duration_ms=400, intensity=1.0, loudness_stats=dune_stats
    )
    assert typo_sub.crest_scale == 0.72


def test_shout_and_screams_scale_up_to_ceiling():
    """
    High-energy vocal peaks and screams scale up to 1.35x - 1.62x and activate box-fit compression.
    """
    dune_stats = AudioLoudnessStats(
        median_speech_rms_db=-33.0,
        whisper_rms_db=-43.0,
        shout_rms_db=-16.0,
        deadband_db=2.0,
    )

    # Full shout at -15 dBFS with high prominence
    prosody_shout = ProsodyMetrics(
        rms_db=-15.0,
        f0_word_hz=210.0,
        f0_speaker_median_hz=120.0,
        prominence_score=1.0,
        isolated_silence_hold=False,
    )
    typo_shout = compute_typography_spec(
        prosody_shout, duration_ms=600, intensity=1.0, loudness_stats=dune_stats
    )
    assert 1.45 <= typo_shout.crest_scale <= 1.62
    # Ascender/descender compression check for vertical box fit
    assert typo_shout.ytas < 750
    assert typo_shout.ytde > -203


def test_monotonic_scaling_across_dynamic_range():
    """
    Crest scale must increase monotonically as acoustic loudness increases.
    """
    stats = AudioLoudnessStats(
        median_speech_rms_db=-30.0,
        whisper_rms_db=-42.0,
        shout_rms_db=-14.0,
        deadband_db=1.5,
    )
    db_levels = np.linspace(-45.0, -10.0, 36)
    scales = []
    for db in db_levels:
        prosody = ProsodyMetrics(
            rms_db=float(db),
            f0_word_hz=140.0,
            f0_speaker_median_hz=140.0,
            prominence_score=0.3,
            isolated_silence_hold=False,
        )
        typo = compute_typography_spec(
            prosody, duration_ms=400, intensity=1.0, loudness_stats=stats
        )
        scales.append(typo.crest_scale)

    # Monotonicity check
    for i in range(len(scales) - 1):
        assert scales[i] <= scales[i + 1] + 1e-4, f"Monotonicity violation at index {i}: {scales[i]} > {scales[i+1]}"


def test_intensity_slider_with_adaptive_normalization():
    """
    Intensity slider interpolates correctly between static (1.00) and kinetic scale.
    """
    stats = AudioLoudnessStats(
        median_speech_rms_db=-28.0,
        whisper_rms_db=-40.0,
        shout_rms_db=-12.0,
    )
    prosody_loud = ProsodyMetrics(
        rms_db=-12.0,
        f0_word_hz=180.0,
        f0_speaker_median_hz=130.0,
        prominence_score=0.9,
        isolated_silence_hold=False,
    )
    # At intensity 0.0: static 1.00x
    typo_static = compute_typography_spec(
        prosody_loud, duration_ms=400, intensity=0.0, loudness_stats=stats
    )
    assert typo_static.crest_scale == 1.00

    # At intensity 0.5: halfway between 1.00 and kinetic peak
    typo_half = compute_typography_spec(
        prosody_loud, duration_ms=400, intensity=0.5, loudness_stats=stats
    )
    typo_full = compute_typography_spec(
        prosody_loud, duration_ms=400, intensity=1.0, loudness_stats=stats
    )
    expected_half = round(1.0 + 0.5 * (typo_full.crest_scale - 1.0), 4)
    assert abs(typo_half.crest_scale - expected_half) <= 0.01


def test_compute_audio_loudness_stats_from_series():
    """
    Verifies that compute_audio_loudness_stats accurately determines percentiles and handles silence.
    """
    # Synthetic speech distribution centered at -32 dBFS with whispers around -42 and shouts around -18
    np.random.seed(42)
    synthetic_speech = np.random.normal(loc=-32.0, scale=6.0, size=1000).tolist()
    # Add some silence frames
    synthetic_speech.extend([-75.0, -80.0, -65.0, -90.0] * 50)

    stats = compute_audio_loudness_stats(synthetic_speech, silence_floor_db=-55.0)
    assert -34.0 <= stats.median_speech_rms_db <= -30.0
    assert stats.whisper_rms_db < stats.median_speech_rms_db - 5.0
    assert stats.shout_rms_db > stats.median_speech_rms_db + 6.0


def test_speech_loudness_tracker_streaming():
    """
    Verifies that SpeechLoudnessTracker dynamically updates its baseline as speech frames arrive.
    """
    tracker = SpeechLoudnessTracker(max_history_words=200)

    # Initial state uses default fallback
    init_stats = tracker.get_stats()
    assert init_stats.conversational_median_db == -20.0

    # Feed 50 words from a -32 dBFS dialogue track
    for val in np.random.normal(loc=-32.0, scale=3.0, size=50):
        tracker.update(float(val))

    updated_stats = tracker.get_stats()
    assert -34.0 <= updated_stats.conversational_median_db <= -30.0
    assert updated_stats.whisper_rms_db < -35.0
