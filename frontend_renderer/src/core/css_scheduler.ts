/**
 * Single-Delay CSS Scheduling Helper
 * References:
 *   - docs/00_overview/folders.md
 *   - docs/01_contracts_and_schemas/typography_and_math_constants.md
 *   - docs/07_client_and_rendering/cwi_dom_renderer.md
 */

import { WordToken } from './types';
import { RenderProfile, DEFAULT_CWI_PROFILE, detectSafari } from './profiles';

export interface WordCSSVariables {
  '--cwi-onset-delay-ms': string;
  '--cwi-duration-ms': string;
  '--cwi-speaker-color': string;
  '--cwi-wght': string;
  '--cwi-wdth': string;
  '--cwi-grad': string;
  '--cwi-opsz': string;
  '--cwi-slnt': string;
  '--cwi-xtra': string;
  '--cwi-xopq': string;
  '--cwi-yopq': string;
  '--cwi-ytas': string;
  '--cwi-ytde': string;
  '--cwi-pop-scale': string;
  '--cwi-crest-peak': string;
  '--cwi-crest-sustain': string;
  '--cwi-hold-em': string;
}

/**
 * Pure calculation of the Single-Delay CSS custom property dictionary for a given WordToken,
 * presentation timestamp, and render profile.
 */
export function computeWordCSSVariables(
  word: WordToken,
  presentMs: number,
  profile?: Partial<RenderProfile>
): WordCSSVariables {
  const intensity = Math.max(0.0, Math.min(1.0, profile?.intensity ?? DEFAULT_CWI_PROFILE.intensity));
  const isSafari = profile?.isSafari ?? detectSafari();

  const rawDelayMs = word.start_ms - presentMs;
  const onsetDelayMs = Math.max(420, rawDelayMs);

  const t = word.typography;
  let wght = Math.round(400 + intensity * (t.wght - 400));
  // Safari WebKit wght:400 workaround: renders abnormally thin at 400
  if (isSafari && wght >= 398 && wght <= 402) {
    wght = 404;
  }

  const wdth = Math.round(100 + intensity * (t.wdth - 100));
  const grad = Math.round(intensity * t.grad);
  const popScale = (1.0 + intensity * (t.pop_scale - 1.0)).toFixed(3);
  const crestPeak = (1.0 + intensity * (t.crest_scale - 1.0)).toFixed(4);
  const crestSustain = (1.0 + intensity * (t.crest_sustain_scale - 1.0)).toFixed(4);
  const holdEm = `${(intensity * t.hold_lift_em).toFixed(4)}em`;

  return {
    '--cwi-onset-delay-ms': `${onsetDelayMs}ms`,
    '--cwi-duration-ms': `${word.duration_ms}ms`,
    '--cwi-speaker-color': word.speaker.color_hex,
    '--cwi-wght': `${wght}`,
    '--cwi-wdth': `${wdth}`,
    '--cwi-grad': `${grad}`,
    '--cwi-opsz': `${t.opsz ?? 14}`,
    '--cwi-slnt': `${t.slnt ?? 0}`,
    '--cwi-xtra': `${t.xtra ?? 468}`,
    '--cwi-xopq': `${t.xopq ?? 96}`,
    '--cwi-yopq': `${t.yopq ?? 79}`,
    '--cwi-ytas': `${t.ytas ?? 750}`,
    '--cwi-ytde': `${t.ytde ?? -203}`,
    '--cwi-pop-scale': popScale,
    '--cwi-crest-peak': crestPeak,
    '--cwi-crest-sustain': crestSustain,
    '--cwi-hold-em': holdEm,
  };
}

/**
 * Binds all Single-Delay CSS custom properties directly onto the DOM element on first sight.
 */
export function bindWordCSSVariables(
  el: HTMLElement,
  word: WordToken,
  presentMs: number,
  profile?: Partial<RenderProfile>
): void {
  const vars = computeWordCSSVariables(word, presentMs, profile);
  for (const [key, val] of Object.entries(vars)) {
    el.style.setProperty(key, val);
  }
}
