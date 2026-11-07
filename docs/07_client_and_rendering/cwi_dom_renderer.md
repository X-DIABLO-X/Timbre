# Deterministic DOM/CSS Renderer (`CwiRenderer`), Shadow DOM & Roboto Flex Axes (`cwi_dom_renderer.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint (Section 1.2 & 5.3)](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [Weave (SNU, 2026)](https://xlaude2040.com/weave/), [CwiRenderer](https://stardance.hackclub.com/projects/63123), [Roboto Flex Variable Font Specimen](https://www.csstypestudio.com/library/font/roboto-flex), [Safari Roboto Flex wght:400 Bug Fix](https://stackoverflow.com/questions/74188298/why-does-safari-render-roboto-flex-so-thin-at-400-font-weight), [STEM DHH/Neurodivergent Adaptive Captioning](https://arxiv.org/pdf/2603.15977)

---

## 1. Core Architectural Invariants of `CwiRenderer`

1. **Shadow DOM Isolation (`overlay.js` / `cwi_renderer.ts`):**
   - Mounts an isolated Shadow DOM root (`hostElement.attachShadow({ mode: 'open' })`) over the active `<video>` viewport so host page CSS rules on YouTube, Twitch, or Vimeo can never override caption typography.
2. **Lower 20% Work Area & 90% Black Caption Box (`work_area_box`):**
   - The caption region is strictly confined to the **lower 20% of the video viewport** (`top: 80%; height: 20%; max-height: 20%;`) and renders inside a **90% opacity black backdrop** (`background: rgba(0, 0, 0, 0.90)`) ([CwiRenderer](https://stardance.hackclub.com/projects/63123)).
3. **Dual-Clock Read-Ahead Layer (`-1.75s` Offset & `420 ms` Minimum Floor):**
   - Words enter the DOM in neutral read-ahead white/grey (`#B0B5BC`) **1.75 seconds before** their acoustic onset ($t_{\text{start}}$), with a minimum read-ahead exposure floor of **420 ms** ([Weave](https://xlaude2040.com/weave/)).
4. **Immutable Post-Onset Text Rule:**
   - Once the presentation clock ($t_{\text{present}} = t_{\text{acoustic}} - 1750\text{ ms}$) crosses a word's onset ($t_{\text{present}} \ge \text{word.start\_ms}$), that word node receives `data-cwi-locked="true"` and its `.textContent` and bounding layout geometry are **permanently frozen** ([Weave](https://xlaude2040.com/weave/)). Late ASR/diarization updates may only mutate unlocked words still in the pre-onset read-ahead zone.
5. **Weave's Single-Delay CSS Scheduling Pattern (Zero JS Timer Jitter):**
   - When a word is first created in the read-ahead layer, its onset delay (`--cwi-onset-delay: calc(...)`), spoken duration (`--cwi-duration`), speaker color (`--cwi-speaker-color`), and all [Roboto Flex](https://www.csstypestudio.com/library/font/roboto-flex) axes are bound once as CSS custom properties.
   - A **single CSS `animation-delay`** triggers all 5 kinetic channels simultaneously on the browser's GPU compositor thread at 60–120 fps without `setTimeout` or `requestAnimationFrame` reveal queues ([Weave](https://xlaude2040.com/weave/)).

---

## 2. Complete Scoped Shadow DOM Stylesheet (`extension/content/overlay.css`)

Write the following complete CSS specification to `extension/content/overlay.css`:

```css
/* Bundled Offline Variable Fonts:
 * - RobotoFlex-Full.woff2 (307 KB, all 13 axes)
 * - RobotoFlex-Subset-wdth-wght.woff2 (62 KB subset)
 * - NotoSansKR-Variable.woff2 (Korean Hangul fallback, weight + scale only)
 */
@font-face {
  font-family: 'CwiRobotoFlex';
  src: url('chrome-extension://__MSG_@@extension_id__/assets/fonts/RobotoFlex-Full.woff2') format('woff2-variations');
  font-weight: 100 1000;
  font-stretch: 25% 151%;
  font-display: block;
}

@font-face {
  font-family: 'CwiNotoSansKR';
  src: url('chrome-extension://__MSG_@@extension_id__/assets/fonts/NotoSansKR-Variable.woff2') format('woff2-variations');
  font-weight: 100 900;
  font-display: block;
}

:host {
  all: initial;
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 2147483647;
  overflow: hidden;
  --cwi-vh: 1vh;
}

/* RULE 1: Strictly restricted to the lower 20% work area of the video viewport */
.cwi-work-area {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  top: 80%;
  height: 20%;
  max-height: 20%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 0.5% 4%;
  box-sizing: border-box;
  pointer-events: none;
}

/* RULE 2: 90% opacity black caption box backdrop (rgba(0, 0, 0, 0.90)) */
.cwi-caption-box {
  background-color: rgba(0, 0, 0, 0.90);
  border-radius: 8px;
  padding: 0.45em 0.95em;
  max-width: 92%;
  max-height: 96%;
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  justify-content: center;
  gap: 0.28em 0.36em;
  box-sizing: border-box;
  overflow: hidden;
}

/* Base Word Token in Read-Ahead Layer (1.75s before onset, >= 420ms floor) */
.cwi-word {
  position: relative;
  display: inline-block;
  font-family: 'CwiRobotoFlex', 'CwiNotoSansKR', sans-serif;
  /* Normal speech = 5% of viewport height (1.0x baseline) */
  font-size: calc(5 * var(--cwi-vh));
  line-height: 1.15;
  /* Safari WebKit wght:400 bug fix: disable optical sizing auto-override */
  font-optical-sizing: none;

  /* Pre-bound Roboto Flex 13 Parametric Axes */
  font-variation-settings:
    'wght' var(--cwi-wght, 400),
    'wdth' var(--cwi-wdth, 100),
    'GRAD' var(--cwi-grad, 0),
    'opsz' var(--cwi-opsz, 14),
    'slnt' var(--cwi-slnt, 0),
    'XTRA' var(--cwi-xtra, 468),
    'XOPQ' var(--cwi-xopq, 96),
    'YOPQ' var(--cwi-yopq, 79),
    'YTAS' var(--cwi-ytas, 750),
    'YTDE' var(--cwi-ytde, -203),
    'YTUC' 712,
    'YTLC' 514,
    'YTFI' 738;

  /* CHANNEL 1: Colour Turn via linear-gradient background-clip: text wipe */
  background-image: linear-gradient(
    90deg,
    var(--cwi-speaker-color, #4FC3F7) 0%,
    var(--cwi-speaker-color, #4FC3F7) 50%,
    var(--cwi-readahead-color, #B0B5BC) 50.01%,
    var(--cwi-readahead-color, #B0B5BC) 100%
  );
  background-size: 200% 100%;
  background-position: 100% 0%;
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  color: transparent;

  transform-origin: center bottom;
  will-change: transform, background-position, font-variation-settings;

  /* WEAVE SINGLE-DELAY CSS SCHEDULING PATTERN:
   * A single --cwi-onset-delay-ms triggers Colour Turn, 1.15x Pop, Crest envelope, and Hold lift!
   */
  animation-name: cwi-colour-turn-wipe, cwi-pop-crest-hold-envelope;
  animation-duration: var(--cwi-duration-ms, 400ms), var(--cwi-duration-ms, 400ms);
  animation-delay: var(--cwi-onset-delay-ms, 420ms), var(--cwi-onset-delay-ms, 420ms);
  animation-timing-function: linear, cubic-bezier(0.22, 1, 0.36, 1);
  animation-fill-mode: forwards, forwards;
}

/* Korean Locale Override: Noto Sans KR Variable (retains weight & scale, omits wdth) */
.cwi-word[data-locale="ko"] {
  font-family: 'CwiNotoSansKR', sans-serif;
  font-variation-settings: 'wght' var(--cwi-wght, 400);
}

/* CHANNEL 1 KEYFRAMES: Horizontal letter-by-letter wipe from t_start to t_end */
@keyframes cwi-colour-turn-wipe {
  0% {
    background-position: 100% 0%;
  }
  100% {
    background-position: 0% 0%;
  }
}

/* CHANNELS 2, 3 & 5 KEYFRAMES:
 * - At onset (0%-18%): 1.15x Pop impulse combined with Crest peak overshoot + Hold lift (-0.525em)
 * - At sustain (35%-85%): Crest settles to 0.70 of peak (var(--cwi-crest-sustain)) and sustains
 * - At offset (100%): Releases back to rest while keeping post-onset speaker color
 */
@keyframes cwi-pop-crest-hold-envelope {
  0% {
    transform: translateY(0em) scale(1.0);
  }
  18% {
    transform: translateY(var(--cwi-hold-em, 0em))
               scale(calc(var(--cwi-pop-scale, 1.15) * var(--cwi-crest-peak, 1.0)));
  }
  38% {
    transform: translateY(var(--cwi-hold-em, 0em))
               scale(var(--cwi-crest-sustain, 1.0));
  }
  85% {
    transform: translateY(var(--cwi-hold-em, 0em))
               scale(var(--cwi-crest-sustain, 1.0));
  }
  100% {
    transform: translateY(0em) scale(var(--cwi-crest-sustain, 1.0));
  }
}

/* Non-Speech & SFX Badge (<|BGM|>, <|Applause|>, [sighs], [gasps], [laughter]) */
.cwi-sfx-badge {
  font-family: 'CwiRobotoFlex', sans-serif;
  font-size: calc(3.8 * var(--cwi-vh));
  color: #E8EAED;
  font-variation-settings: 'wght' 500, 'wdth' 95, 'slnt' -10;
  padding: 0.1em 0.45em;
  border: 1px solid rgba(255, 255, 255, 0.35);
  border-radius: 4px;
  margin-right: 0.35em;
}
```

---

## 3. Complete TypeScript `CwiRenderer` & Immutable Post-Onset Lock (`frontend_renderer/src/core/cwi_renderer.ts`)

Write the following complete code to `frontend_renderer/src/core/cwi_renderer.ts`:

```typescript
import { CaptionSpec, WordToken } from './types';

export interface UserRenderProfile {
  name: 'default_cwi' | 'hoh_clean' | 'adhd_spotlight' | 'minimal_static';
  intensity: number; // 0.0 (static) to 1.0 (full 5-channel kinetic)
  isSafari: boolean;
}

export class CwiRenderer {
  private shadowRoot: ShadowRoot;
  private workAreaEl: HTMLDivElement;
  private captionBoxEl: HTMLDivElement;
  private wordNodes: Map<string, HTMLSpanElement> = new Map();
  private wordSpecs: Map<string, WordToken> = new Map();
  private lockedWordIds: Set<string> = new Set();
  private profile: UserRenderProfile;
  private presentationOffsetMs: number = 1750;
  private static readonly MIN_READ_AHEAD_FLOOR_MS = 420;

  constructor(hostContainer: HTMLElement, initialProfile?: Partial<UserRenderProfile>) {
    const isSafari =
      typeof navigator !== 'undefined' &&
      /^((?!chrome|android).)*safari/i.test(navigator.userAgent);

    this.profile = {
      name: initialProfile?.name ?? 'default_cwi',
      intensity: initialProfile?.intensity ?? 1.0,
      isSafari: initialProfile?.isSafari ?? isSafari,
    };

    this.shadowRoot =
      hostContainer.shadowRoot ?? hostContainer.attachShadow({ mode: 'open' });

    this.workAreaEl = document.createElement('div');
    this.workAreaEl.className = 'cwi-work-area';

    this.captionBoxEl = document.createElement('div');
    this.captionBoxEl.className = 'cwi-caption-box';
    this.workAreaEl.appendChild(this.captionBoxEl);
    this.shadowRoot.appendChild(this.workAreaEl);
  }

  public getPresentationTimeMs(acousticTimeMs: number): number {
    return acousticTimeMs - this.presentationOffsetMs;
  }

  /**
   * Ingests a CaptionSpec v2.0-cwi packet and reconciles DOM word nodes.
   * Enforces the IMMUTABLE POST-ONSET TEXT RULE:
   * Once presentationClockMs >= word.start_ms, that word's text and DOM layout are frozen.
   */
  public ingestCaptionSpec(spec: CaptionSpec, currentAcousticTimeMs: number): void {
    this.presentationOffsetMs = spec.presentation_offset_ms || 1750;
    const presentMs = this.getPresentationTimeMs(currentAcousticTimeMs);

    for (const word of spec.words) {
      // Check if word has already crossed onset playhead
      if (presentMs >= word.start_ms) {
        this.lockedWordIds.add(word.word_id);
      }

      const existingEl = this.wordNodes.get(word.word_id);
      if (existingEl) {
        // IMMUTABLE TEXT RULE: Never mutate text or geometry if word has crossed onset!
        if (this.lockedWordIds.has(word.word_id)) {
          continue;
        }
        // Still inside the pre-onset read-ahead zone: allow endpoint ASR or speaker color correction
        existingEl.textContent = word.text;
        existingEl.style.setProperty('--cwi-speaker-color', word.speaker.color_hex);
        this.wordSpecs.set(word.word_id, word);
        continue;
      }

      // First sight of this word: create DOM node in read-ahead state and bind CSS properties once
      const span = document.createElement('span');
      span.className = 'cwi-word';
      span.dataset.wordId = word.word_id;
      span.dataset.locale = spec.locale || 'en';
      span.textContent = word.text;

      this.bindSingleDelayCSSProperties(span, word, presentMs);
      this.captionBoxEl.appendChild(span);
      this.wordNodes.set(word.word_id, span);
      this.wordSpecs.set(word.word_id, word);
    }
  }

  /**
   * Weave Single-Delay CSS Scheduling Pattern:
   * Binds all 5 kinetic channels and 13 Roboto Flex axes once on first sight.
   */
  private bindSingleDelayCSSProperties(
    el: HTMLSpanElement,
    word: WordToken,
    presentMs: number
  ): void {
    const k = Math.max(0.0, Math.min(1.0, this.profile.intensity));
    const rawDelayMs = word.start_ms - presentMs;
    const onsetDelayMs = Math.max(CwiRenderer.MIN_READ_AHEAD_FLOOR_MS, rawDelayMs);

    const t = word.typography;
    let wght = Math.round(400 + k * (t.wght - 400));
    // Safari WebKit bug workaround: wght: 400 renders abnormally thin
    if (this.profile.isSafari && wght >= 398 && wght <= 402) {
      wght = 404;
    }

    const wdth = Math.round(100 + k * (t.wdth - 100));
    const grad = Math.round(k * t.grad);
    const popScale = (1.0 + k * (t.pop_scale - 1.0)).toFixed(3);
    const crestPeak = (1.0 + k * (t.crest_scale - 1.0)).toFixed(4);
    const crestSustain = (1.0 + k * (t.crest_sustain_scale - 1.0)).toFixed(4);
    const holdEm = `${(k * t.hold_lift_em).toFixed(4)}em`;

    el.style.setProperty('--cwi-onset-delay-ms', `${onsetDelayMs}ms`);
    el.style.setProperty('--cwi-duration-ms', `${word.duration_ms}ms`);
    el.style.setProperty('--cwi-speaker-color', word.speaker.color_hex);
    el.style.setProperty('--cwi-wght', `${wght}`);
    el.style.setProperty('--cwi-wdth', `${wdth}`);
    el.style.setProperty('--cwi-grad', `${grad}`);
    el.style.setProperty('--cwi-opsz', `${t.opsz ?? 14}`);
    el.style.setProperty('--cwi-slnt', `${t.slnt ?? 0}`);
    el.style.setProperty('--cwi-xtra', `${t.xtra ?? 468}`);
    el.style.setProperty('--cwi-xopq', `${t.xopq ?? 96}`);
    el.style.setProperty('--cwi-yopq', `${t.yopq ?? 79}`);
    el.style.setProperty('--cwi-ytas', `${t.ytas ?? 750}`);
    el.style.setProperty('--cwi-ytde', `${t.ytde ?? -203}`);
    el.style.setProperty('--cwi-pop-scale', popScale);
    el.style.setProperty('--cwi-crest-peak', crestPeak);
    el.style.setProperty('--cwi-crest-sustain', crestSustain);
    el.style.setProperty('--cwi-hold-em', holdEm);
  }

  /**
   * Locks words whose onset has passed and prunes expired utterances (> 4.5s after end_ms).
   */
  public tick(currentAcousticTimeMs: number): void {
    const presentMs = this.getPresentationTimeMs(currentAcousticTimeMs);
    for (const [wordId, word] of this.wordSpecs.entries()) {
      const el = this.wordNodes.get(wordId);
      if (!el) continue;

      if (presentMs >= word.start_ms && !this.lockedWordIds.has(wordId)) {
        this.lockedWordIds.add(wordId);
        el.dataset.cwiLocked = 'true';
      }

      if (presentMs > word.end_ms + 4500) {
        el.remove();
        this.wordNodes.delete(wordId);
        this.wordSpecs.delete(wordId);
      }
    }
  }
}
```
