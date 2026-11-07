/**
 * Deterministic DOM/CSS Renderer (CwiRenderer)
 * References:
 *   - docs/07_client_and_rendering/cwi_dom_renderer.md
 *   - docs/00_overview/folders.md
 *   - docs/01_contracts_and_schemas/typography_and_math_constants.md
 */

import { CaptionSpec, NonSpeechEvent, WordToken } from './types';
import { RenderProfile, DEFAULT_CWI_PROFILE, detectSafari } from './profiles';

export type { UserRenderProfile, RenderProfile } from './profiles';

export class CwiRenderer {
  private hostContainer: HTMLElement;
  private shadowRoot: ShadowRoot;
  private workAreaEl: HTMLDivElement;
  private captionBoxEl: HTMLDivElement;
  private styleEl: HTMLStyleElement;

  private allSpecs: CaptionSpec[] = [];
  private activeSegmentId: string | null = null;
  private wordNodes: Map<string, HTMLSpanElement> = new Map();
  private wordTokens: Map<string, WordToken> = new Map();
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

    // Inject scoped stylesheet with imported variable font
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
   * Set all available CaptionSpecs for VOD playback.
   * Sorts segments chronologically to support clean read-ahead windowing.
   */
  public setCaptionSpecs(specs: CaptionSpec[]): void {
    this.allSpecs = [...specs].sort((a, b) => {
      const aStart = a.words?.[0]?.start_ms ?? a.non_speech_events?.[0]?.start_ms ?? 0;
      const bStart = b.words?.[0]?.start_ms ?? b.non_speech_events?.[0]?.start_ms ?? 0;
      return aStart - bStart;
    });
  }

  /**
   * Ingests a single CaptionSpec v2.0-cwi packet (e.g. from WebSocket or file).
   */
  public ingestCaptionSpec(spec: CaptionSpec, currentAcousticTimeMs?: number): void {
    const idx = this.allSpecs.findIndex((s) => s.segment_id === spec.segment_id);
    if (idx >= 0) {
      this.allSpecs[idx] = spec;
    } else {
      this.allSpecs.push(spec);
      this.allSpecs.sort((a, b) => {
        const aStart = a.words?.[0]?.start_ms ?? a.non_speech_events?.[0]?.start_ms ?? 0;
        const bStart = b.words?.[0]?.start_ms ?? b.non_speech_events?.[0]?.start_ms ?? 0;
        return aStart - bStart;
      });
    }

    if (spec.presentation_offset_ms) {
      this.presentationOffsetMs = spec.presentation_offset_ms;
    }

    if (currentAcousticTimeMs != null) {
      this.tick(currentAcousticTimeMs);
    }
  }

  /**
   * Finds the best active CaptionSpec for the current playback timestamp.
   * Ensures that only ONE active utterance/sentence occupies the lower 20% box at any given time.
   */
  private findActiveSpec(timeMs: number): CaptionSpec | null {
    if (this.allSpecs.length === 0) return null;

    const offset = this.presentationOffsetMs;
    const lingerMs = 600;

    let candidate: CaptionSpec | null = null;

    for (let i = 0; i < this.allSpecs.length; i++) {
      const spec = this.allSpecs[i];
      const words = spec.words || [];
      const events = spec.non_speech_events || [];
      if (words.length === 0 && events.length === 0) continue;

      const segStart = words.length > 0 ? words[0].start_ms : events[0].start_ms;
      const segEnd = words.length > 0 ? words[words.length - 1].end_ms : events[events.length - 1].end_ms;
      const readAheadOnset = segStart - offset;
      const displayEnd = segEnd + lingerMs;

      if (timeMs >= readAheadOnset && timeMs <= displayEnd) {
        if (!candidate) {
          candidate = spec;
        } else {
          // If previous candidate has completed speech and this spec is upcoming, switch to the upcoming one
          const candWords = candidate.words || [];
          const candEnd = candWords.length > 0 ? candWords[candWords.length - 1].end_ms : 0;
          if (timeMs > candEnd) {
            candidate = spec;
          }
        }
      }
    }

    return candidate;
  }

  /**
   * Switches the active DOM segment cleanly:
   * Removes previous words, mounts new words in read-ahead neutral state, and unhides the caption box.
   */
  private switchActiveSegment(spec: CaptionSpec | null): void {
    if (!spec) {
      if (this.activeSegmentId !== null) {
        this.clearActiveSegment();
      }
      return;
    }

    if (spec.segment_id === this.activeSegmentId) {
      return;
    }

    this.clearActiveSegment();
    this.activeSegmentId = spec.segment_id;

    this.captionBoxEl.style.display = 'inline-flex';
    this.captionBoxEl.style.opacity = '1';

    // 1. Mount Non-Speech Events (<|BGM|>, [laughter], [gasps], etc.)
    for (const ev of spec.non_speech_events || []) {
      const sfxKey = `${ev.type}_${ev.start_ms}_${ev.end_ms}`;
      const badge = document.createElement('span');
      badge.className = 'cwi-sfx-badge';
      badge.dataset.sfxKey = sfxKey;
      badge.dataset.startMs = `${ev.start_ms}`;
      badge.dataset.endMs = `${ev.end_ms}`;
      badge.textContent = ev.label;
      this.captionBoxEl.appendChild(badge);
      this.sfxNodes.set(sfxKey, badge);
    }

    // 2. Mount Word Tokens in Read-Ahead state (-1.75s)
    for (const word of spec.words || []) {
      const span = document.createElement('span');
      span.className = 'cwi-word';
      span.dataset.wordId = word.word_id;
      span.dataset.locale = spec.locale || 'en';
      span.textContent = word.text;

      span.style.setProperty('--cwi-speaker-color', word.speaker.color_hex);
      span.style.setProperty('--cwi-readahead-color', '#B0B5BC');

      this.captionBoxEl.appendChild(span);
      this.wordNodes.set(word.word_id, span);
      this.wordTokens.set(word.word_id, word);
    }
  }

  private clearActiveSegment(): void {
    this.captionBoxEl.innerHTML = '';
    this.captionBoxEl.style.display = 'none';
    this.captionBoxEl.style.opacity = '0';
    this.wordNodes.clear();
    this.wordTokens.clear();
    this.sfxNodes.clear();
    this.lockedWordIds.clear();
    this.activeSegmentId = null;
  }

  /**
   * Frame-by-frame kinetic typography tick:
   * Synchronously reconciles active segments and drives all 5 kinetic channels:
   *   1. Colour Turn wipe (linear-gradient text-clip background-position)
   *   2. 1.15x Pop impulse
   *   3. Crest volume scaling
   *   4. Weight pitch variation (Roboto Flex wght 340..900)
   *   5. Hold lift (-0.525em for isolated silence holds)
   */
  public tick(currentTimeMs: number): void {
    const activeSpec = this.findActiveSpec(currentTimeMs);
    this.switchActiveSegment(activeSpec);

    if (!activeSpec) {
      return;
    }

    const k = Math.max(0.0, Math.min(1.0, this.profile.intensity));

    // Update SFX badges
    for (const [, badge] of this.sfxNodes.entries()) {
      const startMs = Number(badge.dataset.startMs || 0);
      const endMs = Number(badge.dataset.endMs || 0);
      if (currentTimeMs >= startMs - 200 && currentTimeMs <= endMs + 400) {
        badge.style.display = 'inline-block';
        badge.style.opacity = '1';
      } else {
        badge.style.opacity = '0.35';
      }
    }

    // Update Word Tokens across all 5 Kinetic Channels
    for (const [wordId, word] of this.wordTokens.entries()) {
      const el = this.wordNodes.get(wordId);
      if (!el) continue;

      const wordStart = word.start_ms;
      const wordEnd = word.end_ms;
      const wordDur = Math.max(1, word.duration_ms);
      const t = word.typography;

      if (currentTimeMs < wordStart) {
        // -------------------------------------------------------------
        // STATE 1: Dual-Clock Read-Ahead Layer (-1.75s to 0s)
        // -------------------------------------------------------------
        el.style.backgroundPosition = '100% 0%'; // 100% neutral grey #B0B5BC
        el.style.transform = 'translateY(0em) scale(1.0)';
        el.style.fontVariationSettings = "'wght' 400, 'wdth' 100, 'GRAD' 0, 'opsz' 14";
        delete el.dataset.cwiLocked;
      } else if (currentTimeMs <= wordEnd) {
        // -------------------------------------------------------------
        // STATE 2: Active Spoken Utterance
        // -------------------------------------------------------------
        // IMMUTABLE POST-ONSET RULE: Freeze text content & bounding geometry
        if (!this.lockedWordIds.has(wordId)) {
          this.lockedWordIds.add(wordId);
          el.dataset.cwiLocked = 'true';
        }

        const progress = Math.max(0.0, Math.min(1.0, (currentTimeMs - wordStart) / wordDur));

        // CHANNEL 1: Colour Turn wipe (smooth left-to-right color wipe)
        if (this.profile.enableColorTurn) {
          const wipePos = Math.max(0, Math.min(100, (1.0 - progress) * 100));
          el.style.backgroundPosition = `${wipePos.toFixed(2)}% 0%`;
        } else {
          el.style.backgroundPosition = '0% 0%';
        }

        // CHANNELS 2 & 3: 1.15x Pop impulse + Crest volume envelope
        const popVal = this.profile.enablePop ? (t.pop_scale ?? 1.15) : 1.0;
        const crestPeak = this.profile.enableCrest ? (t.crest_scale ?? 1.0) : 1.0;
        const crestSustain = this.profile.enableCrest ? (t.crest_sustain_scale ?? 1.0) : 1.0;

        const sPeak = 1.0 + k * ((popVal * crestPeak) - 1.0);
        const sSustain = 1.0 + k * (crestSustain - 1.0);

        let scale = 1.0;
        if (progress <= 0.18) {
          const pNorm = progress / 0.18;
          scale = 1.0 + pNorm * (sPeak - 1.0);
        } else if (progress <= 0.38) {
          const pNorm = (progress - 0.18) / 0.20;
          scale = sPeak - pNorm * (sPeak - sSustain);
        } else {
          scale = sSustain;
        }

        // CHANNEL 5: Hold Lift (-0.525em for isolated silence hold)
        let holdEm = 0;
        if (this.profile.enableHold && word.prosody?.isolated_silence_hold) {
          holdEm = k * (t.hold_lift_em ?? -0.525);
        }

        el.style.transform = `translateY(${holdEm.toFixed(4)}em) scale(${scale.toFixed(4)})`;

        // CHANNEL 4: Weight Pitch Variation & Variable Axes (Roboto Flex)
        let wght = this.profile.enableWeight ? Math.round(400 + k * (t.wght - 400)) : 400;
        if (this.profile.isSafari && wght >= 398 && wght <= 402) {
          wght = 404;
        }
        const wdth = Math.round(100 + k * ((t.wdth ?? 100) - 100));
        const grad = Math.round(k * (t.grad ?? 0));

        el.style.fontVariationSettings = `'wght' ${wght}, 'wdth' ${wdth}, 'GRAD' ${grad}, 'opsz' ${t.opsz ?? 14}, 'slnt' ${t.slnt ?? 0}, 'XTRA' ${t.xtra ?? 468}, 'XOPQ' ${t.xopq ?? 96}, 'YOPQ' ${t.yopq ?? 79}, 'YTAS' ${t.ytas ?? 750}, 'YTDE' ${t.ytde ?? -203}`;
      } else {
        // -------------------------------------------------------------
        // STATE 3: Post-Onset Spoken Word (Completed)
        // -------------------------------------------------------------
        if (!this.lockedWordIds.has(wordId)) {
          this.lockedWordIds.add(wordId);
          el.dataset.cwiLocked = 'true';
        }

        // Fully wiped to speaker color
        el.style.backgroundPosition = '0% 0%';

        const crestSustain = this.profile.enableCrest ? (t.crest_sustain_scale ?? 1.0) : 1.0;
        const sSustain = 1.0 + k * (crestSustain - 1.0);
        el.style.transform = `translateY(0em) scale(${sSustain.toFixed(4)})`;

        let wght = this.profile.enableWeight ? Math.round(400 + k * (t.wght - 400)) : 400;
        if (this.profile.isSafari && wght >= 398 && wght <= 402) {
          wght = 404;
        }
        const wdth = Math.round(100 + k * ((t.wdth ?? 100) - 100));
        const grad = Math.round(k * (t.grad ?? 0));

        el.style.fontVariationSettings = `'wght' ${wght}, 'wdth' ${wdth}, 'GRAD' ${grad}, 'opsz' ${t.opsz ?? 14}, 'slnt' ${t.slnt ?? 0}, 'XTRA' ${t.xtra ?? 468}, 'XOPQ' ${t.xopq ?? 96}, 'YOPQ' ${t.yopq ?? 79}, 'YTAS' ${t.ytas ?? 750}, 'YTDE' ${t.ytde ?? -203}`;
      }
    }
  }

  public clear(): void {
    this.clearActiveSegment();
    this.allSpecs = [];
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
@import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:opsz,slnt,wdth,wght,GRAD,XOPQ,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC@8..144,-10..0,25..151,100..1000,-200..150,27..175,323..603,25..135,649..854,-305..-98,538..783,416..570,528..760&display=swap');

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
  padding: 0.5% 3%;
  box-sizing: border-box;
  pointer-events: none;
}

/* RULE 2: 90% opacity black caption box backdrop (rgba(0, 0, 0, 0.90)) */
.cwi-caption-box {
  background-color: rgba(0, 0, 0, 0.90);
  border-radius: 8px;
  padding: 0.38em 0.85em;
  max-width: 90%;
  max-height: 94%;
  display: none;
  flex-wrap: wrap;
  align-items: baseline;
  justify-content: center;
  gap: 0.22em 0.35em;
  box-sizing: border-box;
  overflow: hidden;
  text-align: center;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.6);
  transition: opacity 0.15s ease-out;
}

/* Base Word Token in Read-Ahead Layer (1.75s before onset, >= 420ms floor) */
.cwi-word {
  position: relative;
  display: inline-block;
  font-family: 'Roboto Flex', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: calc(4.2 * var(--cwi-vh, 1vh));
  line-height: 1.22;
  font-optical-sizing: none;
  white-space: pre;

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
}

/* Korean Locale Override */
.cwi-word[data-locale="ko"] {
  font-family: 'Noto Sans KR', 'CwiNotoSansKR', sans-serif;
}

/* Non-Speech SFX Badge */
.cwi-sfx-badge {
  font-family: 'Roboto Flex', sans-serif;
  font-size: calc(3.2 * var(--cwi-vh, 1vh));
  color: #E8EAED;
  font-variation-settings: 'wght' 500, 'wdth' 95, 'slnt' -10;
  padding: 0.1em 0.4em;
  border: 1px solid rgba(255, 255, 255, 0.35);
  border-radius: 4px;
  margin-right: 0.35em;
  display: inline-block;
  align-self: center;
  background-color: rgba(255, 255, 255, 0.08);
}
    `.trim();
  }
}
