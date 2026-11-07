/**
 * CaptionSpec v2.0-cwi Data Contract & TypeScript Interfaces
 * Reference: docs/01_contracts_and_schemas/caption_spec_schema.md
 */

export type SpeakerState = 'unknown' | 'provisional' | 'stable' | 'corrected';

export type NonSpeechEventType =
  | 'BGM'
  | 'APPLAUSE'
  | 'LAUGHTER'
  | 'CRYING'
  | 'SIGH'
  | 'GASP'
  | 'COUGH'
  | 'SNEEZE'
  | 'BREATHING'
  | 'SCREAM'
  | 'YAWN'
  | 'THROAT_CLEARING'
  | 'SNIFF';

export type NonSpeechEventLabel =
  | '<|BGM|>'
  | '<|Applause|>'
  | '[laughter]'
  | '[crying]'
  | '[sighs]'
  | '[gasps]'
  | '[coughing]'
  | '[sneezing]'
  | '[breathing]'
  | '[screaming]'
  | '[yawning]'
  | '[throat clearing]'
  | '[sniffs]';

export interface WorkAreaBox {
  readonly bottom_pct: 20;
  readonly bg_opacity: 0.90;
}

export interface NonSpeechEvent {
  type: NonSpeechEventType;
  label: NonSpeechEventLabel;
  start_ms: number;
  end_ms: number;
  confidence?: number;
}

export interface SpeakerInfo {
  id: string;
  state: SpeakerState;
  color_hex: string;
  compass_bearing_deg?: number | null;
  compass_spread_deg?: number | null;
}

export interface ProsodyMetrics {
  rms_db: number;
  f0_word_hz: number;
  f0_speaker_median_hz: number;
  prominence_score: number;
  isolated_silence_hold: boolean;
  pitch_contour_norm?: number[];
}

export interface TypographySpec {
  crest_scale: number;         // [0.72, 1.62]
  crest_sustain_scale: number; // 0.70 * crest_scale (clamped [0.50, 1.62])
  pop_scale: 1.15;             // Constant 1.15x onset impulse
  wght: number;                // [340, 900]
  wdth: number;                // [25, 151]
  grad: number;                // [-200, 150]
  hold_lift_em: number;        // -0.525 if isolated_silence_hold else 0.0
  slnt?: number;               // [-10, 0]
  opsz?: number;               // [8, 144]
  xtra?: number;               // [323, 603]
  xopq?: number;               // [27, 175]
  yopq?: number;               // [25, 135]
  ytas?: number;               // [649, 854]
  ytde?: number;               // [-305, -98]
}

export interface WordToken {
  word_id: string;
  text: string;
  is_disfluency?: boolean;
  start_ms: number;
  end_ms: number;
  duration_ms: number;
  speaker: SpeakerInfo;
  prosody: ProsodyMetrics;
  typography: TypographySpec;
}

export interface CaptionSpec {
  schema_version: '2.0-cwi';
  segment_id: string;
  stream_acoustic_time_ms?: number;
  locale?: string;
  presentation_offset_ms: number;
  work_area_box: WorkAreaBox;
  non_speech_events: NonSpeechEvent[];
  words: WordToken[];
}
