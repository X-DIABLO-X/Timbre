import { WordToken } from './types';
import { RenderProfile, DEFAULT_CWI_PROFILE } from './profiles';

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
 * Computes all Roboto Flex variable font axes and kinetic scaling targets for a word token.
 * Note: Does NOT clamp onset delay to 420ms during direct video playback so words trigger
 * at exact millisecond alignment with <video>.currentTime.
 */
export function computeWordCSSVariables(
  word: WordToken,
  presentMs: number,
  profile?: Partial<RenderProfile>
): WordCSSVariables {
  const prof: RenderProfile = { ...DEFAULT_CWI_PROFILE, ...profile };
  const k = Math.max(0.0, Math.min(1.0, prof.intensity));

  // Exact delay relative to current presentation time (no artificial 420ms minimum lag in VOD mode)
  const onsetDelayMs = Math.max(0, Math.round(word.start_ms - presentMs));

  const t = word.typography;
  const weightK = prof.enableWeight ? k : 0.0;
  let wght = Math.round(400 + weightK * (t.wght - 400));
  if (prof.isSafari && wght >= 398 && wght <= 402) {
    wght = 404;
  }

  const wdth = Math.round(100 + weightK * (t.wdth - 100));
  const grad = Math.round(weightK * t.grad);

  const popScale = prof.enablePop ? 1.0 + k * (t.pop_scale - 1.0) : 1.0;
  const crestPeak = prof.enableCrest ? 1.0 + k * (t.crest_scale - 1.0) : 1.0;
  const crestSustain = prof.enableCrest ? 1.0 + k * (t.crest_sustain_scale - 1.0) : 1.0;
  const holdEm = prof.enableHold ? k * t.hold_lift_em : 0.0;

  return {
    '--cwi-onset-delay-ms': `${onsetDelayMs}ms`,
    '--cwi-duration-ms': `${Math.max(80, word.duration_ms)}ms`,
    '--cwi-speaker-color': prof.enableColorTurn ? word.speaker.color_hex : '#FFFFFF',
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
    '--cwi-pop-scale': popScale.toFixed(4),
    '--cwi-crest-peak': crestPeak.toFixed(4),
    '--cwi-crest-sustain': crestSustain.toFixed(4),
    '--cwi-hold-em': `${holdEm.toFixed(4)}em`,
  };
}

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
