/**
 * Deterministic DOM/CSS Renderer (CwiRenderer)
 * References:
 *   - docs/07_client_and_rendering/cwi_dom_renderer.md
 *   - docs/00_overview/folders.md
 *   - docs/01_contracts_and_schemas/typography_and_math_constants.md
 */

import { CaptionSpec, NonSpeechEvent, WordToken } from './types';
import { RenderProfile, DEFAULT_CWI_PROFILE, detectSafari } from './profiles';
import { bindWordCSSVariables } from './css_scheduler';

export { UserRenderProfile, RenderProfile } from './profiles';

export class CwiRenderer {
  private hostContainer: HTMLElement;
  private shadowRoot: ShadowRoot;
  private workAreaEl: HTMLDivElement;
  private captionBoxEl: HTMLDivElement;
  private styleEl: HTMLStyleElement;
  private wordNodes: Map<string, HTMLSpanElement> = new Map();
  private wordSpecs: Map<string, WordToken> = new Map();
  private sfxNodes: Map<string, HTMLSpanElement> = new Map();
  private lockedWordIds: Set<string> = new Set();
  private profile: RenderProfile;
  private presentationOffsetMs: number = 1750;
  private static readonly MIN_READ_AHEAD_FLOOR_MS = 420;

  constructor(hostContainer: HTMLElement, initialProfile?: Partial<RenderProfile>) {
    this.hostContainer = hostContainer;
    const isSafari = initialProfile?.isSafari ?? detectSafari();

    this.profile = {
      ...DEFAULT_CWI_PROFILE,
      ...initialProfile,
      isSafari,
      intensity: Math.max(0.0, Math.min(1.0, initialProfile?.intensity ?? DEFAULT_CWI_PROFILE.intensity)),
    };

    // Attach or reuse open Shadow DOM
    this.shadowRoot =
      hostContainer.shadowRoot ?? hostContainer.attachShadow({ mode: 'open' });

    // Inject scoped stylesheet
    this.styleEl = document.createElement('style');
    this.styleEl.textContent = CwiRenderer.getDefaultStylesheet();
    this.shadowRoot.appendChild(this.styleEl);

    // RULE 1: Strictly lower 20% work area
    this.workAreaEl = document.createElement('div');
    this.workAreaEl.className = 'cwi-work-area';

    // RULE 2: 90% opacity black caption box
    this.captionBoxEl = document.createElement('div');
    this.captionBoxEl.className = 'cwi-caption-box';
    this.workAreaEl.appendChild(this.captionBoxEl);
    this.shadowRoot.appendChild(this.workAreaEl);
  }

  public getPresentationTimeMs(acousticTimeMs: number): number {
    return acousticTimeMs - this.presentationOffsetMs;
  }

  public getPresentationOffsetMs(): number {
    return this.presentationOffsetMs;
  }

  public setPresentationOffsetMs(offsetMs: number): void {
    this.presentationOffsetMs = Math.max(CwiRenderer.MIN_READ_AHEAD_FLOOR_MS, Math.min(5000, offsetMs));
  }

  public getProfile(): RenderProfile {
    return { ...this.profile };
  }

  public setProfile(updatedProfile: Partial<RenderProfile>): void {
    this.profile = {
      ...this.profile,
      ...updatedProfile,
      intensity: Math.max(0.0, Math.min(1.0, updatedProfile.intensity ?? this.profile.intensity)),
    };
  }

  public setViewportHeightPx(heightPx: number): void {
    const vhPx = heightPx / 100;
    this.hostContainer.style.setProperty('--cwi-vh', `${vhPx}px`);
  }

  /**
   * Ingests a CaptionSpec v2.0-cwi packet and reconciles DOM word nodes.
   * Enforces the IMMUTABLE POST-ONSET TEXT RULE:
   * Once presentationClockMs >= word.start_ms, that word's text and DOM layout are frozen.
   */
  public ingestCaptionSpec(spec: CaptionSpec, currentAcousticTimeMs: number): void {
    this.presentationOffsetMs = spec.presentation_offset_ms || 1750;
    const presentMs = this.getPresentationTimeMs(currentAcousticTimeMs);

    // Reconcile Non-Speech Events (<|BGM|>, <|Applause|>, [laughter], etc.)
    for (const ev of spec.non_speech_events || []) {
      const sfxKey = `${ev.type}_${ev.start_ms}_${ev.end_ms}`;
      if (!this.sfxNodes.has(sfxKey) && presentMs <= ev.end_ms) {
        const sfxBadge = document.createElement('span');
        sfxBadge.className = 'cwi-sfx-badge';
        sfxBadge.dataset.sfxKey = sfxKey;
        sfxBadge.dataset.endMs = `${ev.end_ms}`;
        sfxBadge.textContent = ev.label;
        this.captionBoxEl.insertBefore(sfxBadge, this.captionBoxEl.firstChild);
        this.sfxNodes.set(sfxKey, sfxBadge);
      }
    }

    // Reconcile Words
    for (const word of spec.words || []) {
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
        // Still inside pre-onset read-ahead zone: allow text/speaker corrections
        existingEl.textContent = word.text;
        existingEl.style.setProperty('--cwi-speaker-color', word.speaker.color_hex);
        this.wordSpecs.set(word.word_id, word);
        continue;
      }

      // First sight of this word: create DOM node in read-ahead state and bind Single-Delay CSS properties once
      const span = document.createElement('span');
      span.className = 'cwi-word';
      span.dataset.wordId = word.word_id;
      span.dataset.locale = spec.locale || 'en';
      span.textContent = word.text;

      bindWordCSSVariables(span, word, presentMs, this.profile);

      this.captionBoxEl.appendChild(span);
      this.wordNodes.set(word.word_id, span);
      this.wordSpecs.set(word.word_id, word);
    }
  }

  /**
   * Locks words whose onset has passed and prunes expired utterances (> 4.5s after end_ms).
   */
  public tick(currentAcousticTimeMs: number): void {
    const presentMs = this.getPresentationTimeMs(currentAcousticTimeMs);

    // Prune expired non-speech badges
    for (const [key, node] of this.sfxNodes.entries()) {
      const endMs = Number(node.dataset.endMs || 0);
      if (presentMs > endMs + 3000) {
        node.remove();
        this.sfxNodes.delete(key);
      }
    }

    // Lock and prune words
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
        this.lockedWordIds.delete(wordId);
      }
    }
  }

  public clear(): void {
    this.captionBoxEl.innerHTML = '';
    this.wordNodes.clear();
    this.wordSpecs.clear();
    this.sfxNodes.clear();
    this.lockedWordIds.clear();
  }

  public destroy(): void {
    this.clear();
    this.workAreaEl.remove();
    this.styleEl.remove();
  }

  public isWordLocked(wordId: string): boolean {
    return this.lockedWordIds.has(wordId);
  }

  public getWordCount(): number {
    return this.wordNodes.size;
  }

  public static getDefaultStylesheet(): string {
    return `
:host {
  all: initial;
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 2147483647;
  overflow: hidden;
  --cwi-vh: 1vh;
}

/* RULE 1: Strictly restricted to lower 20% work area of video viewport */
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
  font-family: 'Roboto Flex', 'CwiRobotoFlex', -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: calc(5 * var(--cwi-vh, 1vh));
  line-height: 1.15;
  font-optical-sizing: none;

  /* Pre-bound Roboto Flex Parametric Axes */
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
    'YTDE' var(--cwi-ytde, -203);

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

  /* WEAVE SINGLE-DELAY CSS SCHEDULING PATTERN */
  animation-name: cwi-colour-turn-wipe, cwi-pop-crest-hold-envelope;
  animation-duration: var(--cwi-duration-ms, 400ms), var(--cwi-duration-ms, 400ms);
  animation-delay: var(--cwi-onset-delay-ms, 420ms), var(--cwi-onset-delay-ms, 420ms);
  animation-timing-function: linear, cubic-bezier(0.22, 1, 0.36, 1);
  animation-fill-mode: forwards, forwards;
}

/* Korean Locale Override */
.cwi-word[data-locale="ko"] {
  font-family: 'Noto Sans KR', 'CwiNotoSansKR', sans-serif;
  font-variation-settings: 'wght' var(--cwi-wght, 400);
}

/* CHANNEL 1 KEYFRAMES: Horizontal letter-by-letter wipe */
@keyframes cwi-colour-turn-wipe {
  0% {
    background-position: 100% 0%;
  }
  100% {
    background-position: 0% 0%;
  }
}

/* CHANNELS 2, 3 & 5 KEYFRAMES */
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

/* Non-Speech SFX Badge */
.cwi-sfx-badge {
  font-family: 'Roboto Flex', sans-serif;
  font-size: calc(3.8 * var(--cwi-vh, 1vh));
  color: #E8EAED;
  font-variation-settings: 'wght' 500, 'wdth' 95, 'slnt' -10;
  padding: 0.1em 0.45em;
  border: 1px solid rgba(255, 255, 255, 0.35);
  border-radius: 4px;
  margin-right: 0.35em;
  display: inline-block;
  align-self: center;
}
    `.trim();
  }
}
