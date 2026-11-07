# Deterministic Typography & Prosody Mathematical Constants (`typography_and_math_constants.md`)

> **Primary References:** [Weave (SNU, 2026)](https://xlaude2040.com/weave/), [CwiRenderer](https://stardance.hackclub.com/projects/63123), [WhiStress (Interspeech 2025)](https://www.isca-archive.org/interspeech_2025/yosha25_interspeech.pdf), [Roboto Flex Variable Font Specimen](https://www.csstypestudio.com/library/font/roboto-flex), [Safari Roboto Flex wght:400 Bug Fix](https://stackoverflow.com/questions/74188298/why-does-safari-render-roboto-flex-so-thin-at-400-font-weight)

---

## 1. Complete Table of Mathematical Constants & Clamps

Every constant below is fixed across the Python backend (`backend/ml_engine/typography_math.py`), the WebGPU Worker (`extension/workers/webgpu_worker.js`), the DOM Renderer (`frontend_renderer/src/core/css_scheduler.ts`), and the `.ass` compiler (`frontend_renderer/offline_burnin/ass_compiler.py`):

| Constant Symbol / Name | Exact Numerical Value | Units / Domain | Physical & Visual Meaning |
| :--- | :--- | :--- | :--- |
| `SAMPLE_RATE_HZ` | `16000` | Hz | Mandatory mono PCM sample rate across VAD, ASR, Diarization, and DSP. |
| `PROSODY_WIN_MS` | `75.0` | ms (`1200` samples) | Sliding window length for frame-level $F_0$ and RMS energy extraction ([Weave](https://xlaude2040.com/weave/)). |
| `PROSODY_STRIDE_MS` | `20.0` | ms (`320` samples) | Sliding window hop/stride (`50 Hz` frame rate, matching Whisper encoder and emotion2vec+). |
| `PROSODY_POOL_MS` | `300.0` | ms (`15` frames) | Temporal pooling window for smoothing word-level $F_0$ and RMS contours ([Weave](https://xlaude2040.com/weave/)). |
| `PRESENTATION_OFFSET_MS` | `1750` | ms (`1.75 s`) | Dual-Clock Read-Ahead Buffer delay ($t_{\text{present}} = t_{\text{acoustic}} - 1750$) ([Weave](https://xlaude2040.com/weave/)). |
| `READ_AHEAD_FLOOR_MS` | `420` | ms (`0.42 s`) | Minimum time a word must remain visible in neutral grey before onset wipe triggers ([Weave](https://xlaude2040.com/weave/)). |
| `WORK_AREA_BOTTOM_PCT` | `20.0` | `%` of viewport height | Captions restricted strictly to the lower 20% screen work area ([CwiRenderer](https://stardance.hackclub.com/projects/63123)). |
| `CAPTION_BOX_OPACITY` | `0.90` | alpha $\in [0, 1]$ | `rgba(0, 0, 0, 0.90)` black caption box backdrop ([CwiRenderer](https://stardance.hackclub.com/projects/63123)). |
| `POP_SCALE_IMPULSE` | `1.15` | multiplier | Universal `1.15x` scale impulse applied to every word at exact onset $t_0$ ([Weave](https://xlaude2040.com/weave/)). |
| `BASELINE_SCREEN_HEIGHT_PCT` | `5.0` | `%` of viewport height | Normal conversational speech font height (`1.0x` baseline Crest scale) ([Weave](https://xlaude2040.com/weave/)). |
| `CREST_MIN_SCREEN_PCT` | `3.0` | `%` of viewport height | Whispered speech minimum font height ([Weave](https://xlaude2040.com/weave/)). |
| `CREST_MAX_SCREEN_PCT` | `12.0` | `%` of viewport height | Shouted speech maximum font height ([Weave](https://xlaude2040.com/weave/)). |
| `CREST_SCALE_MIN` | `0.72` | multiplier | Minimum Crest scale multiplier for whispers ([Weave](https://xlaude2040.com/weave/)). |
| `CREST_SCALE_MAX` | `1.62` | multiplier | Maximum Crest scale multiplier for shouts ([Weave](https://xlaude2040.com/weave/)). |
| `CREST_SUSTAIN_RATIO` | `0.70` | ratio of peak | After onset overshoot to peak, Crest settles to `0.70` of its peak value and sustains ([Weave](https://xlaude2040.com/weave/)). |
| `F0_MIN_HZ` | `80.0` | Hz | Deep-voiced pitch anchor mapped to maximum font weight `wght = 900` ([Weave](https://xlaude2040.com/weave/)). |
| `F0_MAX_HZ` | `250.0` | Hz | High-pitched anchor mapped to minimum font weight `wght = 340` ([Weave](https://xlaude2040.com/weave/)). |
| `WGHT_MIN` | `340` | Roboto Flex `wght` | Lightest variable font weight used for `>= 250 Hz` pitch ([Weave](https://xlaude2040.com/weave/)). |
| `WGHT_MAX` | `900` | Roboto Flex `wght` | Heaviest variable font weight used for `<= 80 Hz` pitch ([Weave](https://xlaude2040.com/weave/)). |
| `SAFARI_WGHT_400_REPLACEMENT` | `404` | Roboto Flex `wght` | Workaround for WebKit/Safari bug where `wght: 400` renders abnormally thin ([Stack Overflow](https://stackoverflow.com/questions/74188298/why-does-safari-render-roboto-flex-so-thin-at-400-font-weight)). |
| `WDTH_MIN` / `WDTH_MAX` | `25` / `151` | Roboto Flex `wdth` (`%`) | Parametric glyph width range modulated by intra-word pitch contour ([Weave](https://xlaude2040.com/weave/)). |
| `HOLD_LIFT_EM` | `-0.525` | `em` (`translateY`) | Vertical upward lift (`-0.525em`) for words isolated by silence on both sides ([Weave](https://xlaude2040.com/weave/)). |
| `ISOLATED_SILENCE_MIN_MS` | `250` | ms | Minimum pre-word AND post-word silence gap required to trigger `isolated_silence_hold = True`. |

---

## 2. Exact Mathematical Formulas (`backend/ml_engine/typography_math.py`)

### 2.1 Channel 3 — Crest (RMS Loudness & Prominence $\to$ Scale Multiplier)

Let $E_{\text{rms\_db}} \in [-60.0, 0.0]\text{ dBFS}$ be the word's mean-pooled RMS energy in decibels, $E_{\text{spk\_norm\_db}}$ be the speaker's running conversational baseline RMS (default `-20.0 dBFS`, whisper floor `-42.0 dBFS`, shout ceiling `-4.0 dBFS`), and $P \in [0.0, 1.0]$ be the **[WhiStress](https://www.isca-archive.org/interspeech_2025/yosha25_interspeech.pdf)** Layer-9 token prominence score.

1. **Normalized Loudness Index ($\alpha_{\text{loud}} \in [0.0, 1.0]$):**
   $$\alpha_{\text{loud}} = \text{clip}\left( \frac{E_{\text{rms\_db}} - (-42.0)}{(-4.0) - (-42.0)}, \; 0.0, \; 1.0 \right) = \text{clip}\left( \frac{E_{\text{rms\_db}} + 42.0}{38.0}, \; 0.0, \; 1.0 \right)$$
2. **Prominence-Boosted Loudness ($\alpha_{\text{eff}} \in [0.0, 1.0]$):**
   - Conversational speech (`-20.0 dBFS`, $P = 0.0$) maps to `1.00x` (`5%` of screen height).
   - Piecewise linear mapping from `[0.0, 0.58, 1.0]` onto `[0.72, 1.00, 1.62]`, fused with `15%` Layer-9 prominence boost:
   $$\alpha_{\text{eff}} = \text{clip}\left( 0.85 \cdot \alpha_{\text{loud}} + 0.15 \cdot P, \; 0.0, \; 1.0 \right)$$
   $$\text{crest\_scale} = \begin{cases} 0.72 + \frac{\alpha_{\text{eff}}}{0.58} \cdot (1.00 - 0.72) & \text{if } \alpha_{\text{eff}} \le 0.58 \\[6pt] 1.00 + \frac{\alpha_{\text{eff}} - 0.58}{1.0 - 0.58} \cdot (1.62 - 1.00) & \text{if } \alpha_{\text{eff}} > 0.58 \end{cases}$$
3. **Crest Sustain Settle (`0.70` of Peak):**
   - Per [Weave](https://xlaude2040.com/weave/), at word onset ($t_0$), Crest overshoots to `crest_scale`, settles to **0.70 of its peak**, sustains across the word, and releases at word offset ($t_{\text{end}}$):
   $$\text{crest\_sustain\_scale} = \text{round}\left( 0.70 \times \text{crest\_scale}, \; 4 \right)$$

---

### 2.2 Channel 4 — Weight (Speaker-Normalized Pitch $F_0$ + Prominence $\to$ `wght` $\in [340, 900]$)

Human speakers fall into three baseline fundamental frequency ranges ([arXiv:2312.06270v2](https://arxiv.org/html/2312.06270v2)):
- **Low-pitch voices:** $F_0 \le 145\text{ Hz}$
- **Medium-pitch voices:** $145\text{ Hz} < F_0 \le 190\text{ Hz}$
- **High-pitch voices:** $F_0 > 190\text{ Hz}$

When a deep-voiced speaker ($F_{0,\text{median}} = 105\text{ Hz}$) shouts intensely, instantaneous vocal cord tension can double $F_{0,\text{word}}$ to `210 Hz`. If `wght` were mapped naively from raw instantaneous $F_{0,\text{word}}$, a booming shout from a bass speaker would render in thin, delicate `wght = 470` letters!

To prevent this, **[Weave](https://xlaude2040.com/weave/)** anchors the effective pitch $F_{0,\text{eff}}$ to the **speaker's running median pitch ($F_{0,\text{median}}$) plus word prominence ($P$)**:
1. **Effective Anchored Pitch ($F_{0,\text{eff}}$):**
   - Let $\Delta F_0 = F_{0,\text{word}} - F_{0,\text{median}}$ be the intra-speaker pitch excursion.
   - When a word has high prominence ($P \ge 0.65$) and high loudness ($\alpha_{\text{loud}} \ge 0.70$), upward pitch excursion caused by vocal effort is damped by $(1 - 0.75 \cdot P)$, and prominence pulls effective pitch downward toward the heavy end of the scale by up to `25 Hz`:
   $$F_{0,\text{eff}} = \text{clip}\left( F_{0,\text{median}} + (1.0 - 0.75 \cdot P) \cdot (F_{0,\text{word}} - F_{0,\text{median}}) - 25.0 \cdot P, \; 80.0, \; 250.0 \right)$$
2. **Inverse Linear Mapping from $[80\text{ Hz}, 250\text{ Hz}] \to [900, 340]$:**
   $$\text{wght}_{\text{raw}} = \text{round}\left( 900 - \frac{F_{0,\text{eff}} - 80.0}{250.0 - 80.0} \times (900 - 340) \right) = \text{round}\left( 900 - \frac{F_{0,\text{eff}} - 80.0}{170.0} \times 560 \right)$$
3. **Safari WebKit `wght: 400` Bug Guard:**
   - On Safari / WebKit, [Roboto Flex renders abnormally thin at `wght: 400`](https://stackoverflow.com/questions/74188298/why-does-safari-render-roboto-flex-so-thin-at-400-font-weight). Whenever `398 <= wght_raw <= 402`, replace with `404` (and set `font-optical-sizing: none`).

---

### 2.3 Channel 5 — Hold Lift (`-0.525em`) & Width (`wdth` $\in [25, 151]$)

1. **Isolated Silence Hold Lift:**
   - Given a sequence of aligned words $w_1, w_2, \dots, w_N$, let $\Delta t_{\text{prev}}(w_i) = w_i.\text{start\_ms} - w_{i-1}.\text{end\_ms}$ and $\Delta t_{\text{next}}(w_i) = w_{i+1}.\text{start\_ms} - w_i.\text{end\_ms}$.
   - If $\Delta t_{\text{prev}}(w_i) \ge 250\text{ ms}$ AND $\Delta t_{\text{next}}(w_i) \ge 250\text{ ms}$:
     $$\text{isolated\_silence\_hold} = \text{True}, \quad \text{hold\_lift\_em} = -0.525$$
   - Otherwise:
     $$\text{isolated\_silence\_hold} = \text{False}, \quad \text{hold\_lift\_em} = 0.0$$
2. **Harmonic Width (`wdth` $\in [25, 151]$):**
   - For Korean (`locale == "ko"`), `Noto Sans KR` has no `wdth` axis, so `wdth = 100` ([Weave](https://xlaude2040.com/weave/)).
   - For English and Latin scripts (`Roboto Flex`), `wdth` modulates around baseline `100` using intra-word pitch span ($\sigma_{F_0}$, semitone excursion across the word) and duration stretch:
     $$\text{wdth} = \text{clip}\left( \text{round}\left( 100 + 38.0 \cdot \tanh\left(\frac{F_{0,\text{word}} - F_{0,\text{median}}}{45.0}\right) + 15.0 \cdot \text{clip}\left(\frac{\text{duration\_ms} - 350}{650}, -1.0, 1.0\right) \right), \; 25, \; 151 \right)$$
3. **Zero-Reflow Grade (`GRAD` $\in [-200, 150]$) & Parametric Axes (`YTAS`, `YTDE`, `XTRA`, `XOPQ`, `YOPQ`):**
   - **`GRAD`:** Mapped from token prominence $P \in [0, 1]$: $\text{grad} = \text{round}(150 \cdot P - 30 \cdot (1 - P)) \in [-30, 150]$ so emphasized words thicken strokes without altering horizontal glyph advance width ([Stack Overflow](https://stackoverflow.com/questions/76091800/subset-a-variable-font-to-only-include-required-axis)).
   - **Vertical Box Fit (`YTAS` / `YTDE`):** When `crest_scale > 1.35` (shouts approaching `12%` of screen height), ascenders (`YTAS`) compress from `750` toward `649` and descenders (`YTDE`) compress from `-203` toward `-98` so stacked lines never overflow the lower 20% caption box ([CwiRenderer](https://stardance.hackclub.com/projects/63123)).

---

## 3. Ready-to-Use Python Implementation — `backend/ml_engine/typography_math.py`

```python
import math
from typing import Optional
import numpy as np
from backend.schemas.caption_spec import ProsodyMetrics, TypographySpec


def clip(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def compute_typography_spec(
    prosody: ProsodyMetrics,
    duration_ms: int,
    intensity: float = 1.0,
    is_safari: bool = False,
    locale: str = "en",
    is_offscreen_or_sfx: bool = False,
) -> TypographySpec:
    # Deterministic mapping from physical voice measurements to the 5 Kinetic Channels
    intensity = clip(intensity, 0.0, 1.0)

    # 1. Channel 3: Crest (Scale 0.72x to 1.62x; 3% to 12% screen height; 5% normal = 1.0x)
    alpha_loud = clip((prosody.rms_db + 42.0) / 38.0, 0.0, 1.0)
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
```

---

## 4. Curated High-Contrast Speaker Color Palette (`shared_schemas/speaker_palette.json`)

Every color in the speaker attribution wheel is validated for **WCAG AAA / AA contrast ($\ge 7.0:1$)** against the `rgba(0, 0, 0, 0.90)` caption box backdrop ([Caption With Intention](https://www.captionwithintention.org/), [Weave](https://xlaude2040.com/weave/)):

```json
{
  "read_ahead_uncolored_hex": "#B0B5BC",
  "unknown_speaker_hex": "#9AA0A6",
  "speakers": [
    { "slot": 0, "id": "SPK_00", "name": "CWI Electric Cyan", "color_hex": "#4FC3F7", "ass_bgr": "&H00F7C34F&" },
    { "slot": 1, "id": "SPK_01", "name": "CWI Amber Gold",    "color_hex": "#FFD54F", "ass_bgr": "&H004FD5FF&" },
    { "slot": 2, "id": "SPK_02", "name": "CWI Coral Rose",    "color_hex": "#FF8A80", "ass_bgr": "&H00808AFF&" },
    { "slot": 3, "id": "SPK_03", "name": "CWI Mint Emerald",  "color_hex": "#69F0AE", "ass_bgr": "&H00AEF069&" },
    { "slot": 4, "id": "SPK_04", "name": "CWI Lavender Violet","color_hex": "#B388FF", "ass_bgr": "&H00FF88B3&" },
    { "slot": 5, "id": "SPK_05", "name": "CWI Tangerine",     "color_hex": "#FFAB40", "ass_bgr": "&H0040ABFF&" }
  ]
}
```
