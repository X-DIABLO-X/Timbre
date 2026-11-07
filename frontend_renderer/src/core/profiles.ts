/**
 * User Customization Profiles & Safari Workaround Definitions
 * References:
 *   - docs/00_overview/folders.md
 *   - docs/07_client_and_rendering/cwi_dom_renderer.md
 *   - docs/07_client_and_rendering/react_player_and_popup.md
 */

export type ProfileName = 'default_cwi' | 'hoh_clean' | 'adhd_spotlight' | 'minimal_static';

export interface RenderProfile {
  name: ProfileName;
  label: string;
  description: string;
  intensity: number; // 0.0 (static) to 1.0 (full 5-channel kinetic)
  isSafari: boolean;
  enableColorTurn: boolean;
  enablePop: boolean;
  enableCrest: boolean;
  enableWeight: boolean;
  enableHold: boolean;
  enableVoiceCompass: boolean;
}

export type UserRenderProfile = RenderProfile;

export function detectSafari(): boolean {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent;
  return /^((?!chrome|android).)*safari/i.test(ua);
}

export const DEFAULT_CWI_PROFILE: RenderProfile = {
  name: 'default_cwi',
  label: 'Default CWI',
  description: 'Full 5-channel expressive typography with dynamic weight, crest scaling, and hold lift.',
  intensity: 1.0,
  isSafari: detectSafari(),
  enableColorTurn: true,
  enablePop: true,
  enableCrest: true,
  enableWeight: true,
  enableHold: true,
  enableVoiceCompass: true,
};

export const HOH_CLEAN_PROFILE: RenderProfile = {
  name: 'hoh_clean',
  label: 'HoH Clean Separation',
  description: 'Optimized for Hard-of-Hearing viewers with high-contrast speaker separation and muted kinetic motion.',
  intensity: 0.7,
  isSafari: detectSafari(),
  enableColorTurn: true,
  enablePop: false,
  enableCrest: true,
  enableWeight: true,
  enableHold: false,
  enableVoiceCompass: true,
};

export const ADHD_SPOTLIGHT_PROFILE: RenderProfile = {
  name: 'adhd_spotlight',
  label: 'ADHD Prominence Spotlight',
  description: 'Enhanced prominence spotlighting key nouns and stressed words to aid sustained focus.',
  intensity: 1.0,
  isSafari: detectSafari(),
  enableColorTurn: true,
  enablePop: true,
  enableCrest: true,
  enableWeight: true,
  enableHold: true,
  enableVoiceCompass: false,
};

export const MINIMAL_STATIC_PROFILE: RenderProfile = {
  name: 'minimal_static',
  label: 'High-Contrast Minimal',
  description: 'Zero kinetic animation; static subtitles with high-contrast speaker attribution colors.',
  intensity: 0.0,
  isSafari: detectSafari(),
  enableColorTurn: false,
  enablePop: false,
  enableCrest: false,
  enableWeight: false,
  enableHold: false,
  enableVoiceCompass: false,
};

export const PROFILES: Record<ProfileName, RenderProfile> = {
  default_cwi: DEFAULT_CWI_PROFILE,
  hoh_clean: HOH_CLEAN_PROFILE,
  adhd_spotlight: ADHD_SPOTLIGHT_PROFILE,
  minimal_static: MINIMAL_STATIC_PROFILE,
};

export function getProfile(name: ProfileName, overrides?: Partial<RenderProfile>): RenderProfile {
  const base = PROFILES[name] ?? DEFAULT_CWI_PROFILE;
  return {
    ...base,
    ...overrides,
    isSafari: overrides?.isSafari ?? detectSafari(),
    intensity: Math.max(0.0, Math.min(1.0, overrides?.intensity ?? base.intensity)),
  };
}
