/**
 * App.tsx: Standalone React 19 VOD & Live Captioning Studio
 * References:
 *   - docs/00_overview/folders.md
 *   - docs/01_contracts_and_schemas/caption_spec_schema.md
 *   - docs/07_client_and_rendering/react_player_and_popup.md
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { CaptionSpec } from '../core/types';
import { ProfileName, RenderProfile, PROFILES, getProfile } from '../core/profiles';
import { VoiceCompassBearing } from '../core/voice_compass';
import { VideoViewport } from './VideoViewport';
import { ControlPanel } from './ControlPanel';

// Canonical demo CaptionSpec payload from Section 1 of caption_spec_schema.md
const CANONICAL_DEMO_SPECS: CaptionSpec[] = [
  {
    schema_version: '2.0-cwi',
    segment_id: 'seg_0042',
    presentation_offset_ms: 1750,
    work_area_box: {
      bottom_pct: 20,
      bg_opacity: 0.90,
    },
    non_speech_events: [
      {
        type: 'BGM',
        label: '<|BGM|>',
        start_ms: 1000,
        end_ms: 6000,
        confidence: 0.98,
      },
      {
        type: 'APPLAUSE',
        label: '<|Applause|>',
        start_ms: 4500,
        end_ms: 6500,
        confidence: 0.95,
      },
    ],
    words: [
      {
        word_id: 'w_0042_01',
        text: 'THANK',
        start_ms: 2200,
        end_ms: 2680,
        duration_ms: 480,
        speaker: {
          id: 'SPK_01',
          state: 'stable',
          color_hex: '#FFD54F',
          compass_bearing_deg: -35,
          compass_spread_deg: 12,
        },
        prosody: {
          rms_db: -8.4,
          f0_word_hz: 215.0,
          f0_speaker_median_hz: 110.0,
          prominence_score: 0.94,
          isolated_silence_hold: true,
        },
        typography: {
          crest_scale: 1.48,
          crest_sustain_scale: 1.036,
          pop_scale: 1.15,
          wght: 820,
          wdth: 128,
          grad: 85,
          hold_lift_em: -0.525,
        },
      },
      {
        word_id: 'w_0042_02',
        text: 'YOU',
        start_ms: 2750,
        end_ms: 3050,
        duration_ms: 300,
        speaker: {
          id: 'SPK_01',
          state: 'stable',
          color_hex: '#FFD54F',
          compass_bearing_deg: -35,
          compass_spread_deg: 12,
        },
        prosody: {
          rms_db: -18.2,
          f0_word_hz: 125.0,
          f0_speaker_median_hz: 110.0,
          prominence_score: 0.42,
          isolated_silence_hold: false,
        },
        typography: {
          crest_scale: 1.04,
          crest_sustain_scale: 0.728,
          pop_scale: 1.15,
          wght: 520,
          wdth: 98,
          grad: 15,
          hold_lift_em: 0.0,
        },
      },
      {
        word_id: 'w_0042_03',
        text: 'VERY',
        start_ms: 3120,
        end_ms: 3450,
        duration_ms: 330,
        speaker: {
          id: 'SPK_00',
          state: 'stable',
          color_hex: '#4FC3F7',
          compass_bearing_deg: 25,
          compass_spread_deg: 16,
        },
        prosody: {
          rms_db: -14.0,
          f0_word_hz: 155.0,
          f0_speaker_median_hz: 130.0,
          prominence_score: 0.65,
          isolated_silence_hold: false,
        },
        typography: {
          crest_scale: 1.16,
          crest_sustain_scale: 0.812,
          pop_scale: 1.15,
          wght: 640,
          wdth: 108,
          grad: 45,
          hold_lift_em: 0.0,
        },
      },
      {
        word_id: 'w_0042_04',
        text: 'MUCH!',
        start_ms: 3500,
        end_ms: 4100,
        duration_ms: 600,
        speaker: {
          id: 'SPK_00',
          state: 'stable',
          color_hex: '#4FC3F7',
          compass_bearing_deg: 25,
          compass_spread_deg: 16,
        },
        prosody: {
          rms_db: -6.5,
          f0_word_hz: 190.0,
          f0_speaker_median_hz: 130.0,
          prominence_score: 0.98,
          isolated_silence_hold: true,
        },
        typography: {
          crest_scale: 1.55,
          crest_sustain_scale: 1.085,
          pop_scale: 1.15,
          wght: 870,
          wdth: 136,
          grad: 110,
          hold_lift_em: -0.525,
        },
      },
    ],
  },
];

export const App: React.FC = () => {
  const [profileName, setProfileName] = useState<ProfileName>('default_cwi');
  const [intensity, setIntensity] = useState<number>(1.0);
  const [presentationOffsetMs, setPresentationOffsetMs] = useState<number>(1750);
  const [isLiveMode, setIsLiveMode] = useState<boolean>(false);
  const [captionSpecs, setCaptionSpecs] = useState<CaptionSpec[]>(CANONICAL_DEMO_SPECS);
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [currentTimeMs, setCurrentTimeMs] = useState<number>(0);
  const [wsStatus, setWsStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');

  // Derive active RenderProfile
  const activeProfile = useMemo<RenderProfile>(() => {
    return getProfile(profileName, { intensity });
  }, [profileName, intensity]);

  // Derive active voice compass bearing from words near the current playhead
  const activeVoiceBearing = useMemo<VoiceCompassBearing | null>(() => {
    for (const spec of captionSpecs) {
      for (const w of spec.words || []) {
        if (
          currentTimeMs >= w.start_ms - 200 &&
          currentTimeMs <= w.end_ms + 400 &&
          w.speaker.compass_bearing_deg != null
        ) {
          return {
            bearingDeg: w.speaker.compass_bearing_deg,
            spreadDeg: w.speaker.compass_spread_deg ?? 15,
            colorHex: w.speaker.color_hex,
            speakerId: w.speaker.id,
          };
        }
      }
    }
    return null;
  }, [captionSpecs, currentTimeMs]);

  // Live WebSocket Connection Handler
  useEffect(() => {
    if (!isLiveMode) {
      setWsStatus('disconnected');
      return;
    }

    setWsStatus('connecting');
    const ws = new WebSocket('ws://127.0.0.1:8765/ws');

    ws.onopen = () => {
      setWsStatus('connected');
      // Send SESSION_START control frame per Section 5 of caption_spec_schema.md
      ws.send(
        JSON.stringify({
          type: 'SESSION_START',
          session_id: `player_${Date.now()}`,
          sample_rate: 16000,
          channels: 1,
          format: 'pcm_s16le',
          mode: 'live_streaming',
          locale: 'en',
          enable_stem_separation: false,
        })
      );
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.schema_version === '2.0-cwi') {
          setCaptionSpecs((prev) => [...prev.slice(-20), data as CaptionSpec]);
        }
      } catch (e) {
        // Non-JSON or binary frames ignored in client viewer
      }
    };

    ws.onerror = () => {
      setWsStatus('error');
    };

    ws.onclose = () => {
      setWsStatus('disconnected');
    };

    return () => {
      ws.close();
    };
  }, [isLiveMode]);

  const handleLoadSampleData = useCallback(() => {
    setCaptionSpecs(CANONICAL_DEMO_SPECS);
  }, []);

  const handleLoadVideoFile = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    setVideoSrc(url);
  }, []);

  const handleLoadCaptionFile = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const parsed = JSON.parse(e.target?.result as string);
        if (Array.isArray(parsed)) {
          setCaptionSpecs(parsed as CaptionSpec[]);
        } else if (parsed && parsed.schema_version === '2.0-cwi') {
          setCaptionSpecs([parsed as CaptionSpec]);
        } else {
          alert('Invalid CaptionSpec v2.0-cwi JSON file.');
        }
      } catch (err) {
        alert('Failed to parse caption JSON file.');
      }
    };
    reader.readAsText(file);
  }, []);

  return (
    <div
      style={{
        minHeight: '100vh',
        backgroundColor: '#0a0b0d',
        color: '#eff3f4',
        fontFamily: "'Roboto Flex', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: '24px 16px 48px',
        boxSizing: 'border-box',
        gap: '24px',
      }}
    >
      {/* Header */}
      <header
        style={{
          width: '100%',
          maxWidth: '1280px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid #2f3336',
          paddingBottom: '16px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '1.6rem' }}>⚡</span>
          <div>
            <h1 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
              Caption with Intention (CWI)
            </h1>
            <p style={{ margin: 0, fontSize: '0.8rem', color: '#71767b' }}>
              Deterministic 5-Channel Expressive Subtitle Studio & Player (v2.0-cwi)
            </p>
          </div>
        </div>

        {isLiveMode && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem' }}>
            <span
              style={{
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                backgroundColor:
                  wsStatus === 'connected' ? '#00ba7c' : wsStatus === 'connecting' ? '#f59e0b' : '#f4212e',
              }}
            />
            <span style={{ textTransform: 'capitalize', color: '#71767b' }}>
              WS: {wsStatus} (ws://127.0.0.1:8765/ws)
            </span>
          </div>
        )}
      </header>

      {/* Main Video Viewport */}
      <VideoViewport
        videoSrc={videoSrc}
        captionSpecs={captionSpecs}
        profile={activeProfile}
        presentationOffsetMs={presentationOffsetMs}
        voiceBearing={activeVoiceBearing}
        onTimeUpdate={setCurrentTimeMs}
      />

      {/* Interactive Control Panel */}
      <ControlPanel
        intensity={intensity}
        onIntensityChange={setIntensity}
        selectedProfile={profileName}
        onProfileSelect={(p) => {
          setProfileName(p);
          setIntensity(PROFILES[p].intensity);
        }}
        isLiveMode={isLiveMode}
        onLiveModeToggle={setIsLiveMode}
        presentationOffsetMs={presentationOffsetMs}
        onPresentationOffsetChange={setPresentationOffsetMs}
        captionSpecs={captionSpecs}
        onLoadSampleData={handleLoadSampleData}
        onLoadVideoFile={handleLoadVideoFile}
        onLoadCaptionFile={handleLoadCaptionFile}
      />
    </div>
  );
};

export default App;
