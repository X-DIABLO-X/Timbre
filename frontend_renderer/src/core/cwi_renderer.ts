import { CaptionSpec, WordToken } from './types';
import { RenderProfile, DEFAULT_CWI_PROFILE, detectSafari } from './profiles';
import { bindWordCSSVariables } from './css_scheduler';
import { PRESENTATION_OFFSET_MS } from './dual_clock';

type WordRenderPhase = 'unmounted' | 'pre' | 'active' | 'post';

/**
 * Deterministic 60-120 FPS CwiRenderer.
 * Fixes Issue #2 (audio-to-subtitle sync lag & animation stutter):
 *   1. Removes competing CSS @keyframes from .cwi-word so tick(currentTimeMs) has 100%
 *      frame-accurate authority over GPU-composited translate3d(), scale3d(), and background-position.
 *   2. Evaluates word onset directly against currentTimeMs == word.start_ms (zero 420ms or 1750ms onset lag),
 *      while using presentationOffsetMs (1750ms) strictly as the pre-onset Read-Ahead visibility window!
 *   3. Caches each word's render phase ('pre' | 'active' | 'post') so static pre-onset and post-onset
 *      words incur ZERO DOM writes during requestAnimationFrame ticks.
 */
export class CwiRenderer {
  private shadowRoot: ShadowRoot;
  private styleEl: HTMLStyleElement;
  private workAreaEl: HTMLDivElement;
  private captionBoxEl: HTMLDivElement;

  private specs: CaptionSpec[] = [];
  private activeSegmentId: string | null = null;
  private wordNodes: Map<string, HTMLSpanElement> = new Map();
  private charNodes: Map<string, HTMLSpanElement[]> = new Map();
  private wordSpecs: Map<string, WordToken> = new Map();
  private wordPhases: Map<string, WordRenderPhase> = new Map();
  private lockedWordIds: Set<string> = new Set();

  private profile: RenderProfile;
  private presentationOffsetMs: number = PRESENTATION_OFFSET_MS;

  constructor(hostContainer: HTMLElement, initialProfile?: Partial<RenderProfile>) {
    this.profile = {
      ...DEFAULT_CWI_PROFILE,
      isSafari: detectSafari(),
      ...initialProfile,
    };

    this.shadowRoot =
      hostContainer.shadowRoot ?? hostContainer.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = '';

    this.styleEl = document.createElement('style');
    this.styleEl.textContent = CwiRenderer.getDefaultStylesheet();
    this.shadowRoot.appendChild(this.styleEl);

    this.workAreaEl = document.createElement('div');
    this.workAreaEl.className = 'cwi-work-area';

    this.captionBoxEl = document.createElement('div');
    this.captionBoxEl.className = 'cwi-caption-box';
    this.captionBoxEl.style.display = 'none';

    this.workAreaEl.appendChild(this.captionBoxEl);
    this.shadowRoot.appendChild(this.workAreaEl);
  }

  public static getDefaultStylesheet(): string {
    return `
      @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:opsz,slnt,wdth,wght,GRAD,XOPQ,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC@8..144,-10..0,25..151,100..1000,-200..150,27..175,323..603,25..135,649..854,-305..-98,560..788,416..570,528..760&display=swap');

      :host {
        all: initial;
        position: absolute;
        inset: 0;
        pointer-events: none;
        z-index: 2147483647;
        overflow: hidden;
        --cwi-vh: 1vh;
        --cwi-readahead-color: #B0B5BC;
      }

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
        padding: 0.4% 4%;
        box-sizing: border-box;
        pointer-events: none;
      }

      .cwi-caption-box {
        --cwi-readahead-color: #B0B5BC;
        background-color: rgba(0, 0, 0, 0.86);
        border-radius: 6px;
        padding: 0.38em 0.88em;
        max-width: 90%;
        max-height: 96%;
        display: inline-flex;
        flex-wrap: wrap;
        align-items: baseline;
        justify-content: center;
        gap: 0.22em 0.34em;
        box-sizing: border-box;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.60);
        contain: layout style;
      }

      .cwi-word {
        position: relative;
        display: inline-flex;
        align-items: baseline;
        white-space: pre;
        font-family: 'Roboto Flex', system-ui, -apple-system, sans-serif;
        font-size: calc(4.8 * var(--cwi-vh, 1vh));
        line-height: 1.16;
        vertical-align: baseline;
        font-optical-sizing: none;
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
        transform-origin: center bottom;
      }

      .cwi-char {
        display: inline-block;
        vertical-align: baseline;
        transform-origin: center bottom;
        will-change: transform, color;
        color: var(--cwi-readahead-color, #B0B5BC);
        transform: translate3d(0, 0em, 0) scale3d(1, 1, 1);
        backface-visibility: hidden;
        -webkit-font-smoothing: antialiased;
      }

      .cwi-sfx-badge {
        font-family: 'Roboto Flex', sans-serif;
        font-size: calc(3.6 * var(--cwi-vh, 1vh));
        color: #E8EAED;
        font-variation-settings: 'wght' 520, 'wdth' 96, 'slnt' -10;
        padding: 0.1em 0.45em;
        border: 1px solid rgba(255, 255, 255, 0.32);
        border-radius: 5px;
        margin-right: 0.32em;
      }
    `;
  }

  private static smoothstep(edge0: number, edge1: number, x: number): number {
    const t = Math.max(0.0, Math.min(1.0, (x - edge0) / Math.max(1e-5, edge1 - edge0)));
    return t * t * (3.0 - 2.0 * t);
  }

  public getPresentationTimeMs(acousticTimeMs: number): number {
    return acousticTimeMs - this.presentationOffsetMs;
  }

  public getPresentationOffsetMs(): number {
    return this.presentationOffsetMs;
  }

  public setPresentationOffsetMs(offsetMs: number): void {
    this.presentationOffsetMs = Math.max(0, offsetMs);
  }

  public getProfile(): RenderProfile {
    return { ...this.profile };
  }

  public setProfile(updatedProfile: Partial<RenderProfile>): void {
    this.profile = { ...this.profile, ...updatedProfile };
    // Re-bind CSS variables on currently mounted words without rebuilding DOM
    for (const [wordId, word] of this.wordSpecs.entries()) {
      const el = this.wordNodes.get(wordId);
      if (el) {
        bindWordCSSVariables(el, word, word.start_ms, this.profile);
        this.wordPhases.set(wordId, 'unmounted');
      }
    }
  }

  public setViewportHeightPx(heightPx: number): void {
    if (heightPx > 0) {
      const vhPx = heightPx / 100.0;
      this.workAreaEl.style.setProperty('--cwi-vh', `${vhPx.toFixed(3)}px`);
    }
  }

  public setCaptionSpecs(specs: CaptionSpec[]): void {
    this.specs = [...specs].sort((a, b) => {
      const aStart = a.words[0]?.start_ms ?? a.non_speech_events[0]?.start_ms ?? 0;
      const bStart = b.words[0]?.start_ms ?? b.non_speech_events[0]?.start_ms ?? 0;
      return aStart - bStart;
    });
    this.clearActiveSegment();
  }

  public ingestCaptionSpec(spec: CaptionSpec, currentAcousticTimeMs?: number): void {
    const existingIdx = this.specs.findIndex((s) => s.segment_id === spec.segment_id);
    if (existingIdx >= 0) {
      this.specs[existingIdx] = spec;
    } else {
      this.specs.push(spec);
      this.specs.sort((a, b) => {
        const aStart = a.words[0]?.start_ms ?? 0;
        const bStart = b.words[0]?.start_ms ?? 0;
        return aStart - bStart;
      });
    }

    // If this spec is currently mounted or no spec is mounted yet, reconcile live nodes
    if (this.activeSegmentId === spec.segment_id || this.specs.length === 1) {
      const refTimeMs =
        currentAcousticTimeMs !== undefined
          ? this.getPresentationTimeMs(currentAcousticTimeMs)
          : (spec.words[0]?.start_ms ?? 0) - this.presentationOffsetMs;
      this.mountOrUpdateSegment(spec, refTimeMs);
    }
  }

  /**
   * Finds the single active CaptionSpec for currentTimeMs:
   * - Enters Read-Ahead up to presentationOffsetMs (1750ms) before firstWord.start_ms,
   *   clamped so it NEVER preempts the previous segment while the previous segment is still being spoken!
   * - Remains visible until 320ms after lastWord.end_ms (or until the next segment's speech begins).
   */
  private findActiveSpecForTime(currentTimeMs: number): CaptionSpec | null {
    const n = this.specs.length;
    if (n === 0) return null;

    for (let i = 0; i < n; i++) {
      const spec = this.specs[i];
      if (!spec.words || spec.words.length === 0) continue;

      const firstStart = spec.words[0].start_ms;
      const lastEnd = spec.words[spec.words.length - 1].end_ms;

      // Prevent Read-Ahead of segment i from overlapping active speech of segment i-1
      const prevEnd =
        i > 0 && this.specs[i - 1].words.length > 0
          ? this.specs[i - 1].words[this.specs[i - 1].words.length - 1].end_ms + 120
          : 0;

      const readAheadWindowStart = Math.max(prevEnd, firstStart - this.presentationOffsetMs);

      const nextFirstStart =
        i + 1 < n && this.specs[i + 1].words.length > 0
          ? this.specs[i + 1].words[0].start_ms
          : Number.POSITIVE_INFINITY;

      const segmentHoldEnd = Math.min(lastEnd + 360, nextFirstStart);

      if (currentTimeMs >= readAheadWindowStart && currentTimeMs <= segmentHoldEnd) {
        return spec;
      }
    }
    return null;
  }

  private mountOrUpdateSegment(spec: CaptionSpec, currentTimeMs: number): void {
    if (this.activeSegmentId !== spec.segment_id) {
      this.clearActiveSegment();
      this.activeSegmentId = spec.segment_id;
    }

    this.captionBoxEl.style.display = spec.words.length > 0 ? 'flex' : 'none';

    for (const word of spec.words) {
      if (currentTimeMs >= word.start_ms) {
        this.lockedWordIds.add(word.word_id);
      }

      const existingEl = this.wordNodes.get(word.word_id);
      if (existingEl) {
        // Immutable post-onset text rule: never mutate text after vocal onset
        if (this.lockedWordIds.has(word.word_id)) {
          continue;
        }
        if (existingEl.textContent !== word.text) {
          existingEl.innerHTML = '';
          const chars: HTMLSpanElement[] = [];
          for (const ch of word.text) {
            const charSpan = document.createElement('span');
            charSpan.className = 'cwi-char';
            charSpan.textContent = ch;
            existingEl.appendChild(charSpan);
            chars.push(charSpan);
          }
          this.charNodes.set(word.word_id, chars);
        }
        bindWordCSSVariables(existingEl, word, currentTimeMs, this.profile);
        this.wordSpecs.set(word.word_id, word);
        continue;
      }

      const span = document.createElement('span');
      span.className = 'cwi-word';
      span.dataset.wordId = word.word_id;
      span.dataset.locale = spec.locale || 'en';

      const chars: HTMLSpanElement[] = [];
      for (const ch of word.text) {
        const charSpan = document.createElement('span');
        charSpan.className = 'cwi-char';
        charSpan.textContent = ch;
        charSpan.style.color = 'var(--cwi-readahead-color, #B0B5BC)';
        charSpan.style.transform = 'translate3d(0, 0em, 0) scale3d(1, 1, 1)';
        span.appendChild(charSpan);
        chars.push(charSpan);
      }
      this.charNodes.set(word.word_id, chars);

      bindWordCSSVariables(span, word, currentTimeMs, this.profile);
      // Pre-onset neutral typography invariant: words in Read-Ahead start at standard baseline font axes
      if (currentTimeMs < word.start_ms) {
        span.style.setProperty('--cwi-wght', '400');
        span.style.setProperty('--cwi-wdth', '100');
        span.style.setProperty('--cwi-grad', '0');
      }
      this.captionBoxEl.appendChild(span);
      this.wordNodes.set(word.word_id, span);
      this.wordSpecs.set(word.word_id, word);
      this.wordPhases.set(word.word_id, 'unmounted');
    }
  }

  /**
   * High-frequency 60-120 FPS animation tick driven directly by <video>.currentTime * 1000.
   * Only mutates the 1-2 words actively being spoken in the current frame;
   * pre-onset and post-onset words are cached and skipped for zero DOM layout thrashing.
   * Wave kinematics: continuous traveling wave packet propagating left-to-right through
   * individual characters anchored to the bottom baseline.
   */
  public tick(currentTimeMs: number): void {
    const targetSpec = this.findActiveSpecForTime(currentTimeMs);
    if (!targetSpec) {
      if (this.activeSegmentId !== null) {
        this.clearActiveSegment();
      }
      return;
    }

    if (this.activeSegmentId !== targetSpec.segment_id) {
      this.mountOrUpdateSegment(targetSpec, currentTimeMs);
    }

    const k = Math.max(0.0, Math.min(1.0, this.profile.intensity));
    const enableColor = this.profile.enableColorTurn;
    const enablePop = this.profile.enablePop;
    const enableHold = this.profile.enableHold;

    for (const [wordId, word] of this.wordSpecs.entries()) {
      const el = this.wordNodes.get(wordId);
      if (!el) continue;

      const chars = this.charNodes.get(wordId) || [];
      const prevPhase = this.wordPhases.get(wordId);

      // STATE 1: Read-Ahead Layer (currentTimeMs < word.start_ms)
      if (currentTimeMs < word.start_ms) {
        this.lockedWordIds.delete(wordId);
        if (prevPhase !== 'pre') {
          for (let i = 0; i < chars.length; i++) {
            chars[i].style.color = 'var(--cwi-readahead-color, #B0B5BC)';
            chars[i].style.transform = 'translate3d(0, 0em, 0) scale3d(1, 1, 1)';
          }
          el.style.setProperty('--cwi-wght', '400');
          el.style.setProperty('--cwi-wdth', '100');
          el.style.setProperty('--cwi-grad', '0');
          this.wordPhases.set(wordId, 'pre');
        }
        continue;
      }

      if (!this.lockedWordIds.has(wordId)) {
        this.lockedWordIds.add(wordId);
        el.dataset.cwiLocked = 'true';
      }

      const t = word.typography;

      // STATE 3: Post-Onset / Completed Word (currentTimeMs >= word.end_ms)
      if (currentTimeMs >= word.end_ms) {
        if (prevPhase !== 'post') {
          for (let i = 0; i < chars.length; i++) {
            chars[i].style.color = enableColor ? 'var(--cwi-speaker-color, #4FC3F7)' : '#FFFFFF';
            chars[i].style.transform = 'translate3d(0, 0em, 0) scale3d(1, 1, 1)';
          }
          const weightK = this.profile.enableWeight ? k : 0.0;
          let wght = Math.round(400 + weightK * (t.wght - 400));
          if (this.profile.isSafari && wght >= 398 && wght <= 402) {
            wght = 404;
          }
          const wdth = Math.round(100 + weightK * (t.wdth - 100));
          const grad = Math.round(weightK * t.grad);
          el.style.setProperty('--cwi-wght', `${wght}`);
          el.style.setProperty('--cwi-wdth', `${wdth}`);
          el.style.setProperty('--cwi-grad', `${grad}`);
          this.wordPhases.set(wordId, 'post');
        }
        continue;
      }

      // STATE 2: Active Word - Traveling Wave Packet across Characters (00:52 Synchronization)
      const dur = Math.max(80, word.duration_ms);
      const progress = Math.max(0.0, Math.min(1.0, (currentTimeMs - word.start_ms) / dur));
      const numChars = chars.length;
      const sPeak = 1.0 + k * Math.max(0.15, (t.crest_scale * t.pop_scale - 1.0));
      const holdPeakEm = enableHold ? k * t.hold_lift_em : 0.0;
      const sigma = Math.max(0.18, 1.2 / Math.max(1, numChars));

      // Performance Optimization (Bug #4): Only mutate font-variation axes once upon onset transition,
      // avoiding HarfBuzz glyph re-rasterization on every 16ms animation tick.
      if (prevPhase !== 'active') {
        this.wordPhases.set(wordId, 'active');
        const weightK = this.profile.enableWeight ? k : 0.0;
        let wght = Math.round(400 + weightK * (t.wght - 400));
        if (this.profile.isSafari && wght >= 398 && wght <= 402) {
          wght = 404;
        }
        const wdth = Math.round(100 + weightK * (t.wdth - 100));
        const grad = Math.round(weightK * t.grad);
        el.style.setProperty('--cwi-wght', `${wght}`);
        el.style.setProperty('--cwi-wdth', `${wdth}`);
        el.style.setProperty('--cwi-grad', `${grad}`);
      }

      for (let i = 0; i < numChars; i++) {

        const charEl = chars[i];
        const tau = (i + 0.5) / Math.max(1, numChars);
        const dist = tau - progress;
        const wave = Math.exp(-(dist * dist) / (2.0 * sigma * sigma));

        const ch = charEl.textContent || '';
        const isPunct = /^[.,\/#!$%\^&\*;:{}=\-_`~()]$/.test(ch);

        const charScale = isPunct ? 1.0 : (enablePop ? (1.0 + (sPeak - 1.0) * wave) : 1.0);
        const yLiftEm = enableHold ? (holdPeakEm * Math.sin(progress * Math.PI) * wave) : 0.0;

        charEl.style.transform = `translate3d(0, ${yLiftEm.toFixed(4)}em, 0) scale3d(${charScale.toFixed(4)}, ${charScale.toFixed(4)}, 1)`;

        if (enableColor) {
          // Horizontal color wipe strictly triggers as progress reaches character threshold
          if (progress >= (i / Math.max(1, numChars))) {
            charEl.style.color = 'var(--cwi-speaker-color, #4FC3F7)';
          } else {
            charEl.style.color = 'var(--cwi-readahead-color, #B0B5BC)';
          }
        } else {
          charEl.style.color = '#FFFFFF';
        }
      }
    }
  }

  private clearActiveSegment(): void {
    this.captionBoxEl.innerHTML = '';
    this.captionBoxEl.style.display = 'none';
    this.activeSegmentId = null;
    this.charNodes.clear();
    this.wordNodes.clear();
    this.wordSpecs.clear();
    this.wordPhases.clear();
  }

  public clear(): void {
    this.specs = [];
    this.lockedWordIds.clear();
    this.clearActiveSegment();
  }

  public destroy(): void {
    this.clear();
    this.shadowRoot.innerHTML = '';
  }

  public isWordLocked(wordId: string): boolean {
    return this.lockedWordIds.has(wordId);
  }

  public getWordCount(): number {
    return this.wordNodes.size;
  }
}
