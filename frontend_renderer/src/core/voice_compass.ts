/**
 * SVG Voice Compass Ring Renderer
 * References:
 *   - docs/07_client_and_rendering/react_player_and_popup.md (Section 2)
 *   - docs/07_client_and_rendering/wearable_spatial_haptics.md (Section 1.3)
 */

export interface VoiceCompassBearing {
  bearingDeg: number;       // -180 to +180 (0 is straight ahead/top)
  spreadDeg: number;        // 0 to 180 (angular dispersion)
  colorHex: string;         // Active speaker color hex code
  speakerId?: string;       // Speaker ID (e.g. SPK_00)
}

/**
 * Normalizes any angle in degrees into [-180.0, 180.0].
 */
export function normalizeBearingDeg(deg: number): number {
  let normalized = deg % 360;
  if (normalized > 180) normalized -= 360;
  if (normalized <= -180) normalized += 360;
  return normalized;
}

/**
 * Computes the SVG path definition ('d' attribute) for an arc centered at (cx, cy)
 * with given radius, bearing angle (-180..+180 deg, where 0 is 12-o'clock),
 * and angular spread in degrees.
 */
export function computeCompassArcPath(
  cx: number,
  cy: number,
  radius: number,
  bearingDeg: number,
  spreadDeg: number
): string {
  const halfSpread = Math.max(8.0, Math.min(90.0, spreadDeg / 2.0));
  // Convert azimuth (-180..+180 where 0 is top/12-o'clock) to SVG polar radians
  const startAngleRad = ((bearingDeg - halfSpread - 90.0) * Math.PI) / 180.0;
  const endAngleRad = ((bearingDeg + halfSpread - 90.0) * Math.PI) / 180.0;

  const x1 = cx + radius * Math.cos(startAngleRad);
  const y1 = cy + radius * Math.sin(startAngleRad);
  const x2 = cx + radius * Math.cos(endAngleRad);
  const y2 = cy + radius * Math.sin(endAngleRad);

  const largeArcFlag = halfSpread * 2.0 > 180.0 ? 1 : 0;
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

/**
 * Generates an SVG string of the Voice Compass Ring for HUD or standalone overlay.
 */
export function renderVoiceCompassSVG(
  bearing: VoiceCompassBearing,
  size: number = 72
): string {
  const cx = size / 2;
  const cy = size / 2;
  const radius = size * 0.40;
  const strokeWidth = Math.max(2, size * 0.07);
  const arcPath = computeCompassArcPath(cx, cy, radius, bearing.bearingDeg, bearing.spreadDeg);

  return `
<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" fill="none" xmlns="http://www.w3.org/2000/svg" class="cwi-voice-compass">
  <!-- Subtle circular track backdrop -->
  <circle cx="${cx}" cy="${cy}" r="${radius}" stroke="rgba(255, 255, 255, 0.15)" stroke-width="${strokeWidth}" />
  <!-- Center marker -->
  <circle cx="${cx}" cy="${cy}" r="${strokeWidth * 0.8}" fill="${bearing.colorHex}" opacity="0.6" />
  <!-- Directional speaker bearing arc -->
  <path d="${arcPath}" stroke="${bearing.colorHex}" stroke-width="${strokeWidth}" stroke-linecap="round" fill="none" />
</svg>
`.trim();
}
