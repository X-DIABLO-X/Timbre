/**
 * Dual-Clock Read-Ahead Controller
 * References:
 *   - docs/00_overview/folders.md
 *   - docs/06_optimizations/optimizations.md (Section 3.1)
 *   - docs/07_client_and_rendering/cwi_dom_renderer.md
 */

export const PRESENTATION_OFFSET_MS = 1750;
export const MIN_READ_AHEAD_FLOOR_MS = 420;

/**
 * Computes the effective onset delay in milliseconds for a word token arriving at acousticArrivalMs.
 * Enforces the 420ms minimum read-ahead floor so viewers always have time to perceive the neutral text.
 */
export function computeEffectiveOnsetDelayMs(
  wordStartMs: number,
  acousticArrivalMs: number,
  presentationOffsetMs: number = PRESENTATION_OFFSET_MS
): number {
  const presentationClockAtArrival = acousticArrivalMs - presentationOffsetMs;
  const rawReadAheadDuration = wordStartMs - presentationClockAtArrival;
  return Math.max(MIN_READ_AHEAD_FLOOR_MS, rawReadAheadDuration);
}

/**
 * DualClockController manages the presentation playhead running 1.75s behind the acoustic clock,
 * absorbing ML transcription/diarization latency while providing a stable pre-onset read-ahead window.
 */
export class DualClockController {
  private presentationOffsetMs: number;
  private readonly readAheadFloorMs: number;

  constructor(
    presentationOffsetMs: number = PRESENTATION_OFFSET_MS,
    readAheadFloorMs: number = MIN_READ_AHEAD_FLOOR_MS
  ) {
    this.presentationOffsetMs = Math.max(readAheadFloorMs, Math.min(5000, presentationOffsetMs));
    this.readAheadFloorMs = readAheadFloorMs;
  }

  public getPresentationOffsetMs(): number {
    return this.presentationOffsetMs;
  }

  public setPresentationOffsetMs(offsetMs: number): void {
    this.presentationOffsetMs = Math.max(this.readAheadFloorMs, Math.min(5000, offsetMs));
  }

  /**
   * Transforms acoustic stream time (t_acoustic) to visual presentation time (t_present):
   * t_present = t_acoustic - presentationOffsetMs (default -1750ms).
   */
  public getPresentationTimeMs(acousticTimeMs: number): number {
    return acousticTimeMs - this.presentationOffsetMs;
  }

  /**
   * Inverse transformation from presentation clock to acoustic clock.
   */
  public getAcousticTimeMs(presentationTimeMs: number): number {
    return presentationTimeMs + this.presentationOffsetMs;
  }

  /**
   * Checks whether a word is currently in the read-ahead zone:
   * wordStartMs - presentMs <= presentationOffsetMs AND wordStartMs > presentMs.
   */
  public isInReadAheadZone(wordStartMs: number, presentMs: number): boolean {
    const diff = wordStartMs - presentMs;
    return diff <= this.presentationOffsetMs && diff > 0;
  }

  /**
   * Checks whether the presentation playhead has reached or passed the word onset (post-onset).
   * Once post-onset is reached, the word text and geometry become immutable.
   */
  public isPostOnset(wordStartMs: number, presentMs: number): boolean {
    return presentMs >= wordStartMs;
  }

  /**
   * Computes onset delay from current presentation time, respecting the minimum floor.
   */
  public computeOnsetDelayMs(wordStartMs: number, presentMs: number): number {
    const raw = wordStartMs - presentMs;
    return Math.max(this.readAheadFloorMs, raw);
  }
}
