/**
 * VideoViewport: React 19 Video Container with lower 20% CWI Shadow DOM overlay
 * References:
 *   - docs/07_client_and_rendering/cwi_dom_renderer.md
 *   - docs/07_client_and_rendering/react_player_and_popup.md
 *   - docs/07_client_and_rendering/wearable_spatial_haptics.md
 */

import React, { useEffect, useRef, useState } from 'react';
import { CaptionSpec } from '../core/types';
import { CwiRenderer } from '../core/cwi_renderer';
import { RenderProfile } from '../core/profiles';
import { VoiceCompassBearing, computeCompassArcPath } from '../core/voice_compass';

export interface VideoViewportProps {
  videoSrc: string | null;
  captionSpecs: CaptionSpec[];
  profile: RenderProfile;
  presentationOffsetMs: number;
  voiceBearing: VoiceCompassBearing | null;
  onTimeUpdate?: (currentTimeMs: number) => void;
  onLoadedMetadata?: (durationSec: number) => void;
}

export const VideoViewport: React.FC<VideoViewportProps> = ({
  videoSrc,
  captionSpecs,
  profile,
  presentationOffsetMs,
  voiceBearing,
  onTimeUpdate,
  onLoadedMetadata,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayHostRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<CwiRenderer | null>(null);
  const [, setIsPlaying] = useState(false);

  // Initialize and update CwiRenderer instance
  useEffect(() => {
    if (!overlayHostRef.current) return;

    if (!rendererRef.current) {
      rendererRef.current = new CwiRenderer(overlayHostRef.current, profile);
    } else {
      rendererRef.current.setProfile(profile);
    }
    rendererRef.current.setPresentationOffsetMs(presentationOffsetMs);

    if (containerRef.current) {
      rendererRef.current.setViewportHeightPx(containerRef.current.clientHeight);
    }
  }, [profile, presentationOffsetMs]);

  // Adjust --cwi-vh on container resize
  useEffect(() => {
    const updateViewportHeight = () => {
      if (containerRef.current && rendererRef.current) {
        const height = containerRef.current.clientHeight;
        rendererRef.current.setViewportHeightPx(height);
      }
    };

    updateViewportHeight();
    window.addEventListener('resize', updateViewportHeight);
    return () => window.removeEventListener('resize', updateViewportHeight);
  }, []);

  // Set specs when captionSpecs changes
  useEffect(() => {
    if (!rendererRef.current) return;
    rendererRef.current.setCaptionSpecs(captionSpecs);
    const currentMs = videoRef.current ? Math.round(videoRef.current.currentTime * 1000) : 0;
    rendererRef.current.tick(currentMs);
  }, [captionSpecs]);

  // Playhead animation loop (60-120 fps requestAnimationFrame)
  useEffect(() => {
    let animId: number;

    const loop = () => {
      const video = videoRef.current;
      if (video && rendererRef.current) {
        const currentMs = Math.round(video.currentTime * 1000);
        rendererRef.current.tick(currentMs);

        if (onTimeUpdate) {
          onTimeUpdate(currentMs);
        }
      }
      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [onTimeUpdate]);

  const compassArc = voiceBearing
    ? computeCompassArcPath(36, 36, 26, voiceBearing.bearingDeg, voiceBearing.spreadDeg)
    : null;

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        width: '100%',
        maxWidth: '1280px',
        aspectRatio: '16 / 9',
        backgroundColor: '#0a0b0d',
        borderRadius: '12px',
        overflow: 'hidden',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.45)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {/* Video Element */}
      {videoSrc ? (
        <video
          ref={videoRef}
          src={videoSrc}
          controls
          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          onPlay={() => setIsPlaying(true)}
          onPause={() => {
            setIsPlaying(false);
            if (videoRef.current && rendererRef.current) {
              rendererRef.current.tick(Math.round(videoRef.current.currentTime * 1000));
            }
          }}
          onSeeked={() => {
            if (videoRef.current && rendererRef.current) {
              rendererRef.current.tick(Math.round(videoRef.current.currentTime * 1000));
            }
          }}
          onTimeUpdate={() => {
            if (videoRef.current && rendererRef.current) {
              rendererRef.current.tick(Math.round(videoRef.current.currentTime * 1000));
            }
          }}
          onLoadedMetadata={() => {
            if (videoRef.current && onLoadedMetadata) {
              onLoadedMetadata(videoRef.current.duration);
            }
            if (containerRef.current && rendererRef.current) {
              rendererRef.current.setViewportHeightPx(containerRef.current.clientHeight);
            }
          }}
        />
      ) : (
        <div
          style={{
            color: '#71767B',
            textAlign: 'center',
            userSelect: 'none',
          }}
        >
          <div style={{ fontSize: '3rem', marginBottom: '8px' }}>🎬</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 500 }}>No Video Loaded</div>
          <div style={{ fontSize: '0.85rem', marginTop: '4px' }}>
            Load a video file or test captions with sample data below
          </div>
        </div>
      )}

      {/* CWI Overlay Shadow Host */}
      <div
        ref={overlayHostRef}
        id="cwi-shadow-host"
        style={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          zIndex: 20,
        }}
      />

      {/* Spatial Voice Compass Ring (Upper Right HUD) */}
      {profile.enableVoiceCompass && voiceBearing && (
        <div
          style={{
            position: 'absolute',
            top: '16px',
            right: '16px',
            width: '72px',
            height: '72px',
            borderRadius: '50%',
            backgroundColor: 'rgba(0, 0, 0, 0.75)',
            backdropFilter: 'blur(4px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 30,
            pointerEvents: 'none',
          }}
          title={`Speaker ${voiceBearing.speakerId ?? ''} Azimuth: ${voiceBearing.bearingDeg}°`}
        >
          <svg width="72" height="72" viewBox="0 0 72 72">
            <circle cx="36" cy="36" r="26" stroke="rgba(255, 255, 255, 0.15)" strokeWidth="3" />
            <circle cx="36" cy="36" r="3" fill={voiceBearing.colorHex} opacity="0.8" />
            {compassArc && (
              <path
                d={compassArc}
                stroke={voiceBearing.colorHex}
                strokeWidth="4"
                strokeLinecap="round"
                fill="none"
              />
            )}
          </svg>
        </div>
      )}
    </div>
  );
};
