export const PRESENTATION_OFFSET_MS = 1750;
export const MIN_READ_AHEAD_FLOOR_MS = 420;

export class DualClockController {
  private presentationOffsetMs: number;
  private readAheadFloorMs: number;

  constructor(
    presentationOffsetMs: number = PRESENTATION_OFFSET_MS,
    readAheadFloorMs: number = MIN_READ_AHEAD_FLOOR_MS
  ) {
    this.presentationOffsetMs = Math.max(0, presentationOffsetMs);
    this.readAheadFloorMs = Math.max(0, readAheadFloorMs);
  }

  public getPresentationOffsetMs(): number {
    return this.presentationOffsetMs;
  }

  public setPresentationOffsetMs(offsetMs: number): void {
    this.presentationOffsetMs = Math.max(0, offsetMs);
  }

  public getPresentationTimeMs(acousticTimeMs: number): number {
    return acousticTimeMs - this.presentationOffsetMs;
  }

  public getAcousticTimeMs(presentationTimeMs: number): number {
    return presentationTimeMs + this.presentationOffsetMs;
  }

  /**
   * Returns true when a word should be pre-displayed in neutral read-ahead grey
   * (up to presentationOffsetMs before its vocal onset wordStartMs).
   */
  public isInReadAheadZone(wordStartMs: number, presentMs: number): boolean {
    const delta = wordStartMs - presentMs;
    return delta > 0 && delta <= this.presentationOffsetMs;
  }

  public isPostOnset(wordStartMs: number, presentMs: number): boolean {
    return presentMs >= wordStartMs;
  }

  /**
   * Computes onset delay in ms. In VOD/video-sync mode (enforceFloor = false),
   * returns the exact millisecond delta without forcing a 420ms minimum lag.
   */
  public computeOnsetDelayMs(
    wordStartMs: number,
    presentMs: number,
    enforceFloor: boolean = false
  ): number {
    const rawDelay = Math.max(0, wordStartMs - presentMs);
    return enforceFloor ? Math.max(this.readAheadFloorMs, rawDelay) : rawDelay;
  }
}

export function computeEffectiveOnsetDelayMs(
  wordStartMs: number,
  acousticArrivalMs: number,
  presentationOffsetMs: number = PRESENTATION_OFFSET_MS
): number {
  const presentationClockAtArrival = acousticArrivalMs - presentationOffsetMs;
  const rawReadAheadDuration = wordStartMs - presentationClockAtArrival;
  return Math.max(MIN_READ_AHEAD_FLOOR_MS, rawReadAheadDuration);
}
