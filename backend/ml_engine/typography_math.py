import math
from typing import Optional, Sequence
import numpy as np

from backend.schemas.caption_spec import AudioLoudnessStats, ProsodyMetrics, TypographySpec

# Table of Mathematical Constants & Clamps
SAMPLE_RATE_HZ: int = 16000
PROSODY_WIN_MS: float = 75.0
PROSODY_STRIDE_MS: float = 20.0
PROSODY_POOL_MS: float = 300.0
PRESENTATION_OFFSET_MS: int = 1750
READ_AHEAD_FLOOR_MS: int = 420
WORK_AREA_BOTTOM_PCT: float = 20.0
CAPTION_BOX_OPACITY: float = 0.90
POP_SCALE_IMPULSE: float = 1.15
BASELINE_SCREEN_HEIGHT_PCT: float = 5.0
CREST_MIN_SCREEN_PCT: float = 3.0
CREST_MAX_SCREEN_PCT: float = 12.0
CREST_SCALE_MIN: float = 0.72
CREST_SCALE_MAX: float = 1.62
CREST_SUSTAIN_RATIO: float = 0.70
F0_MIN_HZ: float = 80.0
F0_MAX_HZ: float = 250.0
WGHT_MIN: int = 340
WGHT_MAX: int = 900
SAFARI_WGHT_400_REPLACEMENT: int = 404
WDTH_MIN: int = 25
WDTH_MAX: int = 151
HOLD_LIFT_EM: float = -0.525
ISOLATED_SILENCE_MIN_MS: int = 250

# Conversational Loudness Baseline Clamps & Normalization Defaults
DEFAULT_MEDIAN_SPEECH_RMS_DB: float = -20.0
DEFAULT_WHISPER_RMS_DB: float = -42.0
DEFAULT_SHOUT_RMS_DB: float = -4.0
DEFAULT_CONVERSATIONAL_DEADBAND_DB: float = 2.0
DEFAULT_LOUDNESS_STATS: AudioLoudnessStats = AudioLoudnessStats(
    median_speech_rms_db=DEFAULT_MEDIAN_SPEECH_RMS_DB,
    whisper_rms_db=DEFAULT_WHISPER_RMS_DB,
    shout_rms_db=DEFAULT_SHOUT_RMS_DB,
    deadband_db=DEFAULT_CONVERSATIONAL_DEADBAND_DB,
)


def clip(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def compute_audio_loudness_stats(
    rms_values: Sequence[float],
    silence_floor_db: float = -55.0,
    deadband_db: float = 2.0,
) -> AudioLoudnessStats:
    """
    Computes speech/dialogue RMS statistics across an entire video or audio track.
    Anchors conversational speech median to 1.00x crest scale, with 15th percentile
    defining the whisper floor (0.72x) and 95th percentile defining the shout ceiling (1.62x).
    """
    valid = [float(r) for r in rms_values if r > silence_floor_db]
    if not valid:
        return DEFAULT_LOUDNESS_STATS
    arr = np.array(valid, dtype=np.float32)
    med = float(np.median(arr))
    p15 = float(np.percentile(arr, 15))
    p85 = float(np.percentile(arr, 85))
    p95 = float(np.percentile(arr, 95))
    whisper = min(p15, med - 6.0)
    shout = max(p95, med + 8.0)
    return AudioLoudnessStats(
        median_speech_rms_db=round(med, 2),
        whisper_rms_db=round(whisper, 2),
        shout_rms_db=round(shout, 2),
        deadband_db=round(deadband_db, 2),
        p15_rms_db=round(p15, 2),
        p85_rms_db=round(p85, 2),
        p95_rms_db=round(p95, 2),
    )


def compute_typography_spec(
    prosody: ProsodyMetrics,
    duration_ms: int,
    intensity: float = 1.0,
    is_safari: bool = False,
    locale: str = "en",
    is_offscreen_or_sfx: bool = False,
    loudness_stats: Optional[AudioLoudnessStats] = None,
) -> TypographySpec:
    # Deterministic mapping from physical voice measurements to the 5 Kinetic Channels
    intensity = clip(intensity, 0.0, 1.0)

    # 1. Channel 3: Crest (Scale 0.72x to 1.62x; 3% to 12% screen height; 5% normal = 1.0x)
    # Per-video adaptive normalization anchors conversational speech to raw_crest = 1.00x,
    # preventing dialogue in quieter mixes (e.g. Dune at -33 dBFS) from shrinking to whispers.
    stats = loudness_stats or DEFAULT_LOUDNESS_STATS
    med_db = stats.median_speech_rms_db
    whisper_db = stats.whisper_rms_db
    shout_db = stats.shout_rms_db
    deadband = max(0.0, stats.deadband_db)

    # Conversational baseline anchor: at normal dialogue volume with baseline prominence,
    # alpha_eff == 0.58 so raw_crest == 1.00 (neither shrunken to whisper nor blown up to shout).
    ALPHA_MEDIAN = 0.58 / 0.85  # ~0.68235

    if prosody.rms_db < med_db - deadband:
        span_quiet = max(1.0, (med_db - deadband) - whisper_db)
        ratio = clip((prosody.rms_db - whisper_db) / span_quiet, 0.0, 1.0)
        alpha_loud = ALPHA_MEDIAN * ratio
    elif prosody.rms_db > med_db + deadband:
        span_loud = max(1.0, shout_db - (med_db + deadband))
        ratio = clip((prosody.rms_db - (med_db + deadband)) / span_loud, 0.0, 1.0)
        alpha_loud = ALPHA_MEDIAN + (1.0 - ALPHA_MEDIAN) * ratio
    else:
        alpha_loud = ALPHA_MEDIAN

    alpha_eff = clip(0.85 * alpha_loud + 0.15 * prosody.prominence_score, 0.0, 1.0)

    if alpha_eff <= 0.58:
        raw_crest = 0.72 + (alpha_eff / 0.58) * (1.00 - 0.72)
    else:
        raw_crest = 1.00 + ((alpha_eff - 0.58) / 0.42) * (1.62 - 1.00)

    # Blend with baseline 1.0 according to user animation intensity slider
    crest_scale = round(clip(1.0 + intensity * (raw_crest - 1.0), 0.72, 1.62), 4)
    crest_sustain_scale = round(clip(0.70 * crest_scale, 0.50, 1.62), 4)

    # 2. Channel 4: Weight (80 Hz -> 900, 250 Hz -> 340, anchored to speaker median F0 + prominence)
    p = prosody.prominence_score
    f0_word = prosody.f0_word_hz if prosody.f0_word_hz > 0 else prosody.f0_speaker_median_hz
    f0_med = prosody.f0_speaker_median_hz

    f0_eff = clip(
        f0_med + (1.0 - 0.75 * p) * (f0_word - f0_med) - 25.0 * p,
        80.0,
        250.0,
    )
    raw_wght = 900.0 - ((f0_eff - 80.0) / 170.0) * 560.0
    wght = int(round(clip(400.0 + intensity * (raw_wght - 400.0), 340.0, 900.0)))

    if is_safari and 398 <= wght <= 402:
        wght = 404

    # 3. Channel 5: Hold (-0.525em lift if isolated by silence on both sides) & Width (wdth 25-151)
    hold_lift_em = round(-0.525 * intensity, 4) if prosody.isolated_silence_hold else 0.0

    if locale.startswith("ko"):
        wdth = 100
    else:
        pitch_term = 38.0 * math.tanh((f0_word - f0_med) / 45.0)
        dur_term = 15.0 * clip((duration_ms - 350.0) / 650.0, -1.0, 1.0)
        raw_wdth = 100.0 + pitch_term + dur_term
        wdth = int(round(clip(100.0 + intensity * (raw_wdth - 100.0), 25.0, 151.0)))

    # 4. Parametric Axes: GRAD, slnt, opsz, XTRA, XOPQ, YOPQ, YTAS, YTDE
    grad = int(round(clip(intensity * (150.0 * p - 30.0 * (1.0 - p)), -200.0, 150.0)))
    slnt = -10.0 if is_offscreen_or_sfx else 0.0
    opsz = round(clip(14.0 * crest_scale, 8.0, 144.0), 1)

    # High-speed dialogue legibility: open internal counters (XTRA) when duration < 180ms
    xtra = 520 if duration_ms < 180 else 468

    # Vertical box fit: compress ascenders/descenders when crest_scale > 1.35
    if crest_scale > 1.35:
        shout_ratio = clip((crest_scale - 1.35) / (1.62 - 1.35), 0.0, 1.0)
        ytas = int(round(750 - shout_ratio * (750 - 649)))
        ytde = int(round(-203 + shout_ratio * (-98 - (-203))))
    else:
        ytas = 750
        ytde = -203

    return TypographySpec(
        crest_scale=crest_scale,
        crest_sustain_scale=crest_sustain_scale,
        pop_scale=1.15,
        wght=wght,
        wdth=wdth,
        grad=grad,
        hold_lift_em=hold_lift_em,
        slnt=slnt,
        opsz=opsz,
        xtra=xtra,
        xopq=96,
        yopq=79,
        ytas=ytas,
        ytde=ytde,
    )
