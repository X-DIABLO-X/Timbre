import React, { useEffect, useRef } from 'react';
import { CaptionSpec } from '../core/types';
import { RenderProfile } from '../core/profiles';
import { CwiRenderer } from '../core/cwi_renderer';
import { VoiceCompassBearing, renderVoiceCompassSVG } from '../core/voice_compass';

export interface VideoViewportProps {
  videoSrc: string | null;
  captionSpecs: CaptionSpec[];
  profile: RenderProfile;
  presentationOffsetMs: number;
  voiceBearing: VoiceCompassBearing | null;
  onTimeUpdate?: (currentTimeMs: number) => void;
  onLoadedMetadata?: (durationSec: number) => void;
}

/**
 * VideoViewport with PLL-Locked High-Precision Sub-Frame Media Clock.
 * Eliminates the 15-45ms HTML5 <video>.currentTime staircase jitter so CwiRenderer.tick()
 * receives a strictly monotonic, 60-120Hz smooth timestamp locked to the audio playhead.
 */
export const VideoViewport: React.FC<VideoViewportProps> = ({
  videoSrc,
  captionSpecs,
  profile,
  presentationOffsetMs,
  voiceBearing,
  onTimeUpdate,
  onLoadedMetadata,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayHostRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<CwiRenderer | null>(null);

  // Initialize CwiRenderer once on mount
  useEffect(() => {
    if (!overlayHostRef.current) return;
    const renderer = new CwiRenderer(overlayHostRef.current, profile);
    renderer.setPresentationOffsetMs(presentationOffsetMs);
    renderer.setCaptionSpecs(captionSpecs);
    rendererRef.current = renderer;

    const updateSize = () => {
      if (containerRef.current && rendererRef.current) {
        rendererRef.current.setViewportHeightPx(containerRef.current.clientHeight);
      }
    };
    updateSize();

    const ro = new ResizeObserver(updateSize);
    if (containerRef.current) {
      ro.observe(containerRef.current);
    }

    return () => {
      ro.disconnect();
      renderer.destroy();
      rendererRef.current = null;
    };
  }, []);

  useEffect(() => {
    rendererRef.current?.setProfile(profile);
    if (videoRef.current) {
      rendererRef.current?.tick(videoRef.current.currentTime * 1000.0);
    }
  }, [profile]);

  useEffect(() => {
    rendererRef.current?.setPresentationOffsetMs(presentationOffsetMs);
    if (videoRef.current) {
      rendererRef.current?.tick(videoRef.current.currentTime * 1000.0);
    }
  }, [presentationOffsetMs]);

  useEffect(() => {
    rendererRef.current?.setCaptionSpecs(captionSpecs);
    if (videoRef.current) {
      rendererRef.current?.tick(videoRef.current.currentTime * 1000.0);
    }
  }, [captionSpecs]);

  // High-Precision PLL Media Clock Loop (60-120 FPS zero-staircase interpolation)
  useEffect(() => {
    let rafId: number | null = null;
    let lastRawVideoSec = -1.0;
    let anchorPerfMs = performance.now();
    let anchorMediaMs = 0.0;
    let lastEmittedMediaMs = 0.0;
    let lastUiCallbackMs = 0.0;

    const loop = (nowPerfMs: number) => {
      const v = videoRef.current;
      const renderer = rendererRef.current;

      if (v && renderer) {
        const rawMediaMs = v.currentTime * 1000.0;

        if (v.paused || v.ended || v.seeking) {
          anchorMediaMs = rawMediaMs;
          anchorPerfMs = nowPerfMs;
          lastRawVideoSec = v.currentTime;
          lastEmittedMediaMs = rawMediaMs;
        } else {
          // Whenever the browser updates <video>.currentTime, gently re-anchor PLL clock
          if (v.currentTime !== lastRawVideoSec) {
            lastRawVideoSec = v.currentTime;
            anchorMediaMs = rawMediaMs;
            anchorPerfMs = nowPerfMs;
          }
          const elapsedWallMs = (nowPerfMs - anchorPerfMs) * (v.playbackRate || 1.0);
          const interpolatedMs = anchorMediaMs + Math.min(elapsedWallMs, 65.0);

          // Enforce strict monotonicity during normal forward playback (unless user seeked backward)
          if (Math.abs(rawMediaMs - lastEmittedMediaMs) > 180.0) {
            lastEmittedMediaMs = rawMediaMs;
          } else {
            lastEmittedMediaMs = Math.max(lastEmittedMediaMs, interpolatedMs);
          }
        }

        renderer.tick(lastEmittedMediaMs);

        // Throttle React parent UI time readout to ~15 Hz so React state never slows down 120 FPS overlay
        if (onTimeUpdate && nowPerfMs - lastUiCallbackMs >= 66.0) {
          lastUiCallbackMs = nowPerfMs;
          onTimeUpdate(Math.round(lastEmittedMediaMs));
        }
      }

      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
    };
  }, [onTimeUpdate]);

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        width: '100%',
        aspectRatio: '16 / 9',
        backgroundColor: '#000',
        borderRadius: '12px',
        overflow: 'hidden',
      }}
    >
      {videoSrc ? (
        <video
          ref={videoRef}
          src={videoSrc}
          controls
          playsInline
          onLoadedMetadata={(e) => onLoadedMetadata?.(e.currentTarget.duration)}
          onSeeked={(e) => {
            rendererRef.current?.tick(e.currentTarget.currentTime * 1000.0);
          }}
          style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
        />
      ) : (
        <div
          style={{
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#9AA0A6',
          }}
        >
          Load a video file (e.g. final_video.mp4) and CaptionSpec JSON to preview.
        </div>
      )}

      <div
        ref={overlayHostRef}
        style={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          zIndex: 20,
        }}
      />

      {profile.enableVoiceCompass && voiceBearing && (
        <div
          style={{
            position: 'absolute',
            right: '24px',
            bottom: '22%',
            width: '56px',
            height: '56px',
            pointerEvents: 'none',
            zIndex: 25,
          }}
          dangerouslySetInnerHTML={{ __html: renderVoiceCompassSVG(voiceBearing) }}
        />
      )}
    </div>
  );
};
