import { describe, it, expect } from 'vitest';

describe('CwiRenderer Active Segment & Windowing Invariants', () => {
  const sampleSpecs = [
    {
      schema_version: '2.0-cwi' as const,
      segment_id: 'seg_0014',
      presentation_offset_ms: 1750,
      work_area_box: { bottom_pct: 20, bg_opacity: 0.9 },
      words: [
        {
          word_id: 'w_0014_01',
          text: 'Richer',
          start_ms: 120866,
          end_ms: 121500,
          duration_ms: 634,
          speaker: { id: 'SPK_00', state: 'stable', color_hex: '#4FC3F7' },
          prosody: { rms_db: -18, f0_word_hz: 135, f0_speaker_median_hz: 130, prominence_score: 0.5, isolated_silence_hold: false },
          typography: { crest_scale: 1.0, crest_sustain_scale: 0.7, pop_scale: 1.15, wght: 400, wdth: 100, grad: 0, hold_lift_em: 0, opsz: 14, slnt: 0, xtra: 468, xopq: 96, yopq: 79, ytas: 750, ytde: -203 }
        }
      ]
    },
    {
      schema_version: '2.0-cwi' as const,
      segment_id: 'seg_0015',
      presentation_offset_ms: 1750,
      work_area_box: { bottom_pct: 20, bg_opacity: 0.9 },
      words: [
        {
          word_id: 'w_0015_01',
          text: 'Our',
          start_ms: 165198,
          end_ms: 165417,
          duration_ms: 219,
          speaker: { id: 'SPK_00', state: 'stable', color_hex: '#4FC3F7' },
          prosody: { rms_db: -18, f0_word_hz: 135, f0_speaker_median_hz: 130, prominence_score: 0.5, isolated_silence_hold: false },
          typography: { crest_scale: 1.17, crest_sustain_scale: 0.819, pop_scale: 1.15, wght: 730, wdth: 100, grad: 0, hold_lift_em: 0, opsz: 14, slnt: 0, xtra: 468, xopq: 96, yopq: 79, ytas: 750, ytde: -203 }
        },
        {
          word_id: 'w_0015_02',
          text: 'warriors',
          start_ms: 165417,
          end_ms: 166001,
          duration_ms: 584,
          speaker: { id: 'SPK_00', state: 'stable', color_hex: '#4FC3F7' },
          prosody: { rms_db: -18, f0_word_hz: 135, f0_speaker_median_hz: 130, prominence_score: 0.5, isolated_silence_hold: false },
          typography: { crest_scale: 1.20, crest_sustain_scale: 0.84, pop_scale: 1.15, wght: 712, wdth: 100, grad: 0, hold_lift_em: 0, opsz: 14, slnt: 0, xtra: 468, xopq: 96, yopq: 79, ytas: 750, ytde: -203 }
        },
        {
          word_id: 'w_0015_03',
          text: "couldn't",
          start_ms: 166001,
          end_ms: 166585,
          duration_ms: 584,
          speaker: { id: 'SPK_00', state: 'stable', color_hex: '#4FC3F7' },
          prosody: { rms_db: -18, f0_word_hz: 135, f0_speaker_median_hz: 130, prominence_score: 0.5, isolated_silence_hold: false },
          typography: { crest_scale: 1.08, crest_sustain_scale: 0.756, pop_scale: 1.15, wght: 714, wdth: 100, grad: 0, hold_lift_em: 0, opsz: 14, slnt: 0, xtra: 468, xopq: 96, yopq: 79, ytas: 750, ytde: -203 }
        }
      ]
    }
  ];

  function findActiveSpec(specs: typeof sampleSpecs, timeMs: number, offsetMs: number = 1750) {
    const lingerMs = 600;
    let candidate: typeof sampleSpecs[0] | null = null;
    for (const spec of specs) {
      const words = spec.words || [];
      if (words.length === 0) continue;
      const segStart = words[0].start_ms;
      const segEnd = words[words.length - 1].end_ms;
      const readAheadOnset = segStart - offsetMs;
      const displayEnd = segEnd + lingerMs;
      if (timeMs >= readAheadOnset && timeMs <= displayEnd) {
        if (!candidate) {
          candidate = spec;
        } else {
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

  it('correctly returns null during long silence (e.g. 140s)', () => {
    const active = findActiveSpec(sampleSpecs, 140000);
    expect(active).toBeNull();
  });

  it('correctly enters read-ahead 1.75s before segment 15 onset (163,448ms)', () => {
    const active = findActiveSpec(sampleSpecs, 165198 - 1750);
    expect(active?.segment_id).toBe('seg_0015');
  });

  it('correctly selects segment 15 at 2:46 (166,000ms) without any leakage of seg_0014', () => {
    const active = findActiveSpec(sampleSpecs, 166000);
    expect(active?.segment_id).toBe('seg_0015');
  });

  it('correctly calculates Colour Turn wipe and pop progress at 166,293ms (mid-word for "couldn\'t")', () => {
    const word = sampleSpecs[1].words[2]; // couldn't: 166001 - 166585
    const nowMs = 166293;
    const progress = (nowMs - word.start_ms) / word.duration_ms;
    expect(progress).toBeGreaterThan(0.4);
    expect(progress).toBeLessThan(0.6);
    const wipePos = (1.0 - progress) * 100;
    expect(wipePos).toBeGreaterThan(40);
    expect(wipePos).toBeLessThan(60);
  });
});
