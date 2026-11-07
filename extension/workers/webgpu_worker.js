import { pipeline, env } from '@huggingface/transformers';

env.allowLocalModels = false;
env.useBrowserCache = true;

let asrPipeline = null;
let affectPipeline = null;
let segmentCounter = 0;
let acousticClockMs = 0;
let speechBuffer = new Float32Array(0);
let bufferStartMs = 0;

self.onmessage = async (e) => {
  if (e.data?.type === 'INIT_WEBGPU_PIPELINE') {
    const workletPort = e.ports[0];
    await initModels();
    workletPort.onmessage = (chunkEvent) => {
      const f32 = new Float32Array(chunkEvent.data.pcmFloat32Buffer);
      onAudioChunk(f32, chunkEvent.data.rmsLinear);
    };
  }
};

async function initModels() {
  const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;
  const device = hasWebGPU ? 'webgpu' : 'wasm';
  // Hybrid quantization: fp32 encoder to preserve pitch/mel fidelity, q8/q4f16 decoder
  asrPipeline = await pipeline('automatic-speech-recognition', 'onnx-community/moonshine-base-ONNX', {
    device,
    dtype: {
      encoder_model: 'fp32',
      decoder_model_merged: hasWebGPU ? 'q4f16' : 'q8',
    },
  });
}

async function onAudioChunk(chunkF32, rmsLinear) {
  const chunkMs = Math.round((chunkF32.length / 16000) * 1000);
  acousticClockMs += chunkMs;

  const isSpeech = rmsLinear >= 0.008;
  if (!isSpeech && speechBuffer.length === 0) return;

  if (speechBuffer.length === 0) {
    bufferStartMs = acousticClockMs - chunkMs;
  }

  const merged = new Float32Array(speechBuffer.length + chunkF32.length);
  merged.set(speechBuffer, 0);
  merged.set(chunkF32, speechBuffer.length);
  speechBuffer = merged;

  const bufferedMs = Math.round((speechBuffer.length / 16000) * 1000);
  if (bufferedMs >= 800 || (!isSpeech && bufferedMs >= 250)) {
    const utterance = speechBuffer;
    const startMs = bufferStartMs;
    speechBuffer = new Float32Array(0);
    await runBrowserInference(utterance, startMs);
  }
}

async function runBrowserInference(utteranceF32, startMs) {
  if (!asrPipeline) return;
  const res = await asrPipeline(utteranceF32, { return_timestamps: 'word' });
  const chunks = res.chunks || [{ text: res.text || '', timestamp: [0, utteranceF32.length / 16000] }];

  // Compute local RMS dB
  let sumSq = 0;
  for (let i = 0; i < utteranceF32.length; i++) sumSq += utteranceF32[i] * utteranceF32[i];
  const rmsDb = Math.max(-100, Math.min(0, 20 * Math.log10(Math.max(Math.sqrt(sumSq / utteranceF32.length), 1e-5))));

  const alphaLoud = Math.max(0, Math.min(1, (rmsDb + 42.0) / 38.0));
  const crestScale = Number((0.72 + alphaLoud * (1.62 - 0.72)).toFixed(4));
  const crestSustain = Number((0.70 * crestScale).toFixed(4));

  segmentCounter += 1;
  const words = chunks
    .filter((c) => c.text && c.text.trim().length > 0)
    .map((c, idx) => {
      const wStart = startMs + Math.round((c.timestamp?.[0] || 0) * 1000);
      const wEnd = Math.max(wStart + 120, startMs + Math.round((c.timestamp?.[1] || 0.4) * 1000));
      return {
        word_id: `w_${String(segmentCounter).padStart(4, '0')}_${String(idx + 1).padStart(2, '0')}`,
        text: c.text.trim(),
        start_ms: wStart,
        end_ms: wEnd,
        duration_ms: wEnd - wStart,
        speaker: {
          id: 'SPK_00',
          state: 'stable',
          color_hex: '#4FC3F7',
          compass_bearing_deg: null,
          compass_spread_deg: null,
        },
        prosody: {
          rms_db: Number(rmsDb.toFixed(2)),
          f0_word_hz: 140.0,
          f0_speaker_median_hz: 135.0,
          prominence_score: Number(alphaLoud.toFixed(3)),
          isolated_silence_hold: false,
        },
        typography: {
          crest_scale: crestScale,
          crest_sustain_scale: crestSustain,
          pop_scale: 1.15,
          wght: 680,
          wdth: 105,
          grad: 40,
          hold_lift_em: 0.0,
        },
      };
    });

  if (words.length > 0) {
    self.postMessage({
      type: 'CAPTION_SPEC_EMITTED',
      payload: {
        schema_version: '2.0-cwi',
        segment_id: `seg_${String(segmentCounter).padStart(4, '0')}`,
        presentation_offset_ms: 1750,
        work_area_box: { bottom_pct: 20, bg_opacity: 0.90 },
        non_speech_events: [],
        words,
      },
    });
  }
}
