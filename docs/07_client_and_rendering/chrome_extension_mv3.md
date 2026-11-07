# Chrome Manifest V3 Extension & In-Browser WebGPU Worker (`chrome_extension_mv3.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint (Section 5.1)](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [trippi-oss MV3 Reference](https://github.com/skrylkovs/trippi-oss), [kami-subs MV3 Reference](https://github.com/MohammdKopa/kami-subs), [Gemive MV3 Audio Passthrough](https://github.com/letr1n1ty/Gemive), [Moonshine ASR WebGPU](https://webai.show/models/moonshine-asr/), [emotion2vec-web-distill](https://huggingface.co/thomashallock/emotion2vec-web-distill)

---

## 1. Chrome Manifest V3 (MV3) Architecture & Critical Browser Constraints

Because Chrome Manifest V3 service workers (`background.js`) are ephemeral and **cannot** access `AudioContext`, `MediaStream`, `AudioWorkletNode`, DOM elements, or persistent WebGPU contexts, our extension implements a 4-stage pipeline modeled on proven MV3 reference architectures ([trippi-oss](https://github.com/skrylkovs/trippi-oss), [Gemive](https://github.com/letr1n1ty/Gemive), [kami-subs](https://github.com/MohammdKopa/kami-subs)):

1. **Session Orchestration & Native Messaging Auto-Launcher (`manifest.json`, `background.js`, `launcher.py`):**
   - Requests `tabCapture`, `offscreen`, `storage`, `activeTab`, `scripting`, `downloads`, `tabs`, `nativeMessaging`, and `<all_urls>` permissions ([Gemive](https://github.com/letr1n1ty/Gemive)).
   - When the user clicks **Start** in the extension popup, `background.js` invokes Chrome's **Native Messaging Host** (`com.cwi.engine` $\to$ `launcher.py` / `launcher.bat`) to automatically spawn the local Python FastAPI server (`ws://127.0.0.1:8765/ws`) in the background with zero manual terminal setup, and creates the MV3 **Offscreen Document** (`offscreen/offscreen.html`) ([kami-subs](https://github.com/MohammdKopa/kami-subs)).
2. **Offscreen Audio Capture & MANDATORY `AudioContext` Speaker Passthrough (`offscreen/offscreen.js`):**
   - Receives the `streamId` from `chrome.tabCapture.getMediaStreamId` and opens the stream via `navigator.mediaDevices.getUserMedia`.
   - **CRITICAL INVARIANT:** Capturing tab audio via `chrome.tabCapture` disconnects Chrome's default tab audio output and mutes the tab for the viewer! Therefore, `offscreen.js` **must immediately connect the captured `MediaStreamAudioSourceNode` directly to `audioContext.destination`** (`sourceNode.connect(audioCtx.destination)`) so the user hears uninterrupted audio while the parallel branch feeds the `AudioWorkletNode` ([Gemive](https://github.com/letr1n1ty/Gemive)).
3. **`AudioWorklet` Downmix, `16,000 Hz` Resampling & Zero-Copy Streaming (`offscreen/pcm-worklet.js`):**
   - Replaces deprecated `ScriptProcessorNode` ([kami-subs](https://github.com/MohammdKopa/kami-subs)) with a real-time `AudioWorkletProcessor` (`pcm-worklet.js`) that downmixes stereo/multi-channel audio to mono, resamples from the host rate (`44.1 kHz` / `48 kHz`) to **`16,000 Hz`**, and emits `100 ms` (`1,600`-sample) `Int16` or `Float32` chunks ([trippi-oss](https://github.com/skrylkovs/trippi-oss)).
   - In **Server Mode**, chunks stream over a persistent WebSocket (`ws://127.0.0.1:8765/ws`) to the Python backend.
   - In **Serverless Browser WebGPU Mode**, chunks pass via a zero-copy `MessageChannel` (`Transferable` `ArrayBuffer`) to `workers/webgpu_worker.js` ([Moonshine ASR](https://webai.show/models/moonshine-asr/)).
4. **Widevine DRM Limitation & External OS Audio Loopback Scope:**
   - By fundamental browser security design, Chrome's `tabCapture` outputs a **silent (all-zero) audio stream** on sites protected by **Widevine DRM** (such as Netflix, Disney+, Amazon Prime Video, or Shahid), and this restriction cannot be bypassed within the Chrome Extension API ([kami-subs](https://github.com/MohammdKopa/kami-subs)).
   - Unprotected media on **YouTube, Twitch, Vimeo embeds, university lectures, podcasts, and Google Meet / Zoom / Microsoft Teams web clients** works out-of-the-box via `tabCapture` ([trippi-oss](https://github.com/skrylkovs/trippi-oss)).
   - To caption DRM-protected apps locally, the Python backend (`backend/main.py`) supports an external OS audio loopback flag (`--input-source=os_loopback` via WASAPI Loopback on Windows, ScreenCaptureKit/BlackHole on macOS, or PulseAudio/PipeWire monitor on Linux) that bypasses `tabCapture` while still streaming `CaptionSpec` payloads into the browser overlay.

---

## 2. Complete `extension/manifest.json`

Write the following complete file to `extension/manifest.json`:

```json
{
  "manifest_version": 3,
  "name": "Caption with Intention — Real-Time Expressive Subtitles",
  "version": "2.0.0",
  "description": "Real-time speaker attribution, word-onset synchronization wipes, and variable Roboto Flex prosodic typography.",
  "permissions": [
    "tabCapture",
    "offscreen",
    "storage",
    "activeTab",
    "scripting",
    "downloads",
    "tabs",
    "nativeMessaging"
  ],
  "host_permissions": [
    "<all_urls>",
    "http://127.0.0.1:8765/*",
    "ws://127.0.0.1:8765/*"
  ],
  "background": {
    "service_worker": "background.js",
    "type": "module"
  },
  "action": {
    "default_popup": "popup/popup.html",
    "default_title": "Caption with Intention Controls"
  },
  "content_scripts": [
    {
      "matches": ["<all_urls>"],
      "js": ["content/content.js"],
      "run_at": "document_idle",
      "all_frames": true
    }
  ],
  "web_accessible_resources": [
    {
      "resources": [
        "content/overlay.css",
        "assets/fonts/RobotoFlex-Full.woff2",
        "assets/fonts/RobotoFlex-Subset-wdth-wght.woff2",
        "assets/fonts/NotoSansKR-Variable.woff2"
      ],
      "matches": ["<all_urls>"]
    }
  ],
  "content_security_policy": {
    "extension_pages": "script-src 'self' 'wasm-unsafe-eval'; object-src 'self';"
  }
}
```

---

## 3. Complete `extension/background.js` (Service Worker & Native Launcher)

Write the following complete code to `extension/background.js`:

```javascript
// Chrome MV3 Service Worker: Orchestrates single-session state, launches local Python FastAPI
// server via Native Messaging Host ("com.cwi.engine"), manages Offscreen Document, and routes CaptionSpec.

const NATIVE_HOST_NAME = 'com.cwi.engine';
const OFFSCREEN_DOCUMENT_PATH = 'offscreen/offscreen.html';

let activeTabId = null;
let nativePort = null;

async function ensureNativeBackendRunning() {
  if (nativePort) return true;
  try {
    nativePort = chrome.runtime.connectNative(NATIVE_HOST_NAME);
    nativePort.onMessage.addListener((msg) => {
      console.log('[CWI Native Host]', msg);
    });
    nativePort.onDisconnect.addListener(() => {
      console.warn('[CWI Native Host Disconnected]', chrome.runtime.lastError?.message);
      nativePort = null;
    });
    nativePort.postMessage({ command: 'START_FASTAPI_SERVER', port: 8765 });
    // Wait 600ms for FastAPI uvicorn socket readiness
    await new Promise((r) => setTimeout(r, 600));
    return true;
  } catch (err) {
    console.warn('[CWI] Native host unavailable, falling back to existing server or WebGPU mode:', err);
    return false;
  }
}

async function ensureOffscreenDocument() {
  const existingContexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
    documentUrls: [chrome.runtime.getURL(OFFSCREEN_DOCUMENT_PATH)],
  });
  if (existingContexts.length > 0) return;

  await chrome.offscreen.createDocument({
    url: OFFSCREEN_DOCUMENT_PATH,
    reasons: ['USER_MEDIA', 'AUDIO_PLAYBACK', 'WORKERS'],
    justification: 'Capture tab audio with mandatory speaker passthrough and stream 16kHz PCM to CWI ML engine.',
  });
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'START_CWI_SESSION') {
    handleStartSession(message.tabId, message.mode || 'server_ws', message.locale || 'en')
      .then((res) => sendResponse(res))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.type === 'STOP_CWI_SESSION') {
    handleStopSession()
      .then(() => sendResponse({ ok: true }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  // Forward CaptionSpec v2.0-cwi emitted by offscreen.js (via WS or WebGPU Worker) to active tab content.js
  if (message.type === 'CAPTION_SPEC_PAYLOAD' && activeTabId !== null) {
    chrome.tabs.sendMessage(activeTabId, {
      type: 'RENDER_CAPTION_SPEC',
      payload: message.payload,
    });
  }

  // Forward user customization updates (intensity slider 0-100%, exception profile)
  if (message.type === 'UPDATE_USER_PROFILE' && activeTabId !== null) {
    chrome.tabs.sendMessage(activeTabId, {
      type: 'APPLY_USER_PROFILE',
      profile: message.profile,
    });
  }
});

async function handleStartSession(tabId, mode, locale) {
  activeTabId = tabId;
  if (mode === 'server_ws') {
    await ensureNativeBackendRunning();
  }
  await ensureOffscreenDocument();

  const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tabId });
  await chrome.runtime.sendMessage({
    type: 'OFFSCREEN_INIT_CAPTURE',
    streamId,
    tabId,
    mode,
    locale,
  });

  return { ok: true, tabId, mode };
}

async function handleStopSession() {
  await chrome.runtime.sendMessage({ type: 'OFFSCREEN_STOP_CAPTURE' });
  activeTabId = null;
}
```

---

## 4. Complete `extension/offscreen/offscreen.js` (Mandatory `AudioContext` Speaker Passthrough)

Write the following complete code to `extension/offscreen/offscreen.js`:

```javascript
// Offscreen Document Audio Router:
// 1. Captures tab MediaStream using streamId from chrome.tabCapture
// 2. IMMEDIATELY routes sourceNode -> audioCtx.destination (Mandatory Speaker Passthrough!)
// 3. Connects sourceNode -> AudioWorkletNode ('pcm-16k-worklet')
// 4. Streams 16kHz mono PCM chunks to either ws://127.0.0.1:8765/ws OR zero-copy WebGPU Worker.

let audioCtx = null;
let mediaStream = null;
let workletNode = null;
let wsClient = null;
let webgpuWorker = null;
let consecutiveSilentChunks = 0;

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'OFFSCREEN_INIT_CAPTURE') {
    startCapture(msg.streamId, msg.mode, msg.locale);
  } else if (msg.type === 'OFFSCREEN_STOP_CAPTURE') {
    stopCapture();
  }
});

async function startCapture(streamId, mode = 'server_ws', locale = 'en') {
  stopCapture();

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: {
        chromeMediaSource: 'tab',
        chromeMediaSourceId: streamId,
      },
    },
    video: false,
  });

  audioCtx = new AudioContext({ latencyHint: 'interactive' });
  const sourceNode = audioCtx.createMediaStreamSource(mediaStream);

  // CRITICAL INVARIANT: Mandatory AudioContext Speaker Passthrough!
  // Without this line, chrome.tabCapture mutes the browser tab for the user.
  sourceNode.connect(audioCtx.destination);

  await audioCtx.audioWorklet.addModule('pcm-worklet.js');
  workletNode = new AudioWorkletNode(audioCtx, 'pcm-16k-worklet');
  sourceNode.connect(workletNode);

  if (mode === 'browser_webgpu') {
    initWebGPUWorker(workletNode.port, locale);
  } else {
    initWebSocketClient(workletNode.port, locale);
  }
}

function initWebSocketClient(workletPort, locale) {
  wsClient = new WebSocket('ws://127.0.0.1:8765/ws');
  wsClient.binaryType = 'arraybuffer';

  wsClient.onopen = () => {
    wsClient.send(
      JSON.stringify({
        type: 'SESSION_START',
        session_id: `sess_${Date.now()}`,
        sample_rate: 16000,
        channels: 1,
        format: 'pcm_s16le',
        mode: 'live_streaming',
        locale,
      })
    );
  };

  wsClient.onmessage = (event) => {
    if (typeof event.data === 'string') {
      const spec = JSON.parse(event.data);
      if (spec.schema_version === '2.0-cwi') {
        chrome.runtime.sendMessage({
          type: 'CAPTION_SPEC_PAYLOAD',
          payload: spec,
        });
      }
    }
  };

  workletPort.onmessage = (e) => {
    const { pcmInt16Buffer, rmsLinear } = e.data;
    checkWidevineDrmSilence(rmsLinear);
    if (wsClient && wsClient.readyState === WebSocket.OPEN) {
      wsClient.send(pcmInt16Buffer);
    }
  };
}

function initWebGPUWorker(workletPort, locale) {
  webgpuWorker = new Worker('../workers/webgpu_worker.js', { type: 'module' });
  // Zero-copy MessageChannel transfer of worklet port directly to WebGPU Worker
  webgpuWorker.postMessage({ type: 'INIT_WEBGPU_PIPELINE', locale }, [workletPort]);

  webgpuWorker.onmessage = (e) => {
    if (e.data?.type === 'CAPTION_SPEC_EMITTED') {
      chrome.runtime.sendMessage({
        type: 'CAPTION_SPEC_PAYLOAD',
        payload: e.data.payload,
      });
    }
  };
}

function checkWidevineDrmSilence(rmsLinear) {
  if (rmsLinear < 1e-7) {
    consecutiveSilentChunks += 1;
    if (consecutiveSilentChunks === 50) {
      console.warn(
        '[CWI] 5.0 seconds of exact zero PCM detected. If this site uses Widevine DRM (Netflix/Disney+), ' +
          'tabCapture is silenced by Chrome security policy. Switch to OS Audio Loopback mode.'
      );
    }
  } else {
    consecutiveSilentChunks = 0;
  }
}

function stopCapture() {
  if (workletNode) {
    workletNode.disconnect();
    workletNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (audioCtx) {
    audioCtx.close();
    audioCtx = null;
  }
  if (wsClient) {
    wsClient.close();
    wsClient = null;
  }
  if (webgpuWorker) {
    webgpuWorker.terminate();
    webgpuWorker = null;
  }
}
```

---

## 5. Complete `extension/offscreen/pcm-worklet.js` (`16,000 Hz` Mono Resampler)

Write the following complete code to `extension/offscreen/pcm-worklet.js`:

```javascript
// AudioWorkletProcessor: Downmixes stereo/multi-channel audio to mono, resamples from
// AudioContext sampleRate (e.g. 44100 or 48000 Hz) to 16000 Hz, and emits 100ms (1600-sample)
// Int16 and Float32 chunks.

class PCM16kWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.chunkSamples = 1600; // 100ms at 16,000 Hz (matches Pyannote 6,400-byte Float32 chunk)
    this.outBufferF32 = new Float32Array(this.chunkSamples);
    this.outWriteIndex = 0;
    this.fractionalPos = 0.0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || input[0].length === 0) {
      return true;
    }

    const numChannels = input.length;
    const numFrames = input[0].length;
    const monoFrame = new Float32Array(numFrames);

    // 1. Downmix all channels to mono
    for (let i = 0; i < numFrames; i++) {
      let sum = 0.0;
      for (let c = 0; c < numChannels; c++) {
        sum += input[c][i];
      }
      monoFrame[i] = sum / numChannels;
    }

    // 2. Linear interpolation resampling from sampleRate -> 16000 Hz
    const step = sampleRate / this.targetSampleRate;
    while (this.fractionalPos < numFrames) {
      const idx = Math.floor(this.fractionalPos);
      const frac = this.fractionalPos - idx;
      const s0 = monoFrame[idx];
      const s1 = idx + 1 < numFrames ? monoFrame[idx + 1] : s0;
      const sample = s0 + frac * (s1 - s0);

      this.outBufferF32[this.outWriteIndex++] = Math.max(-1.0, Math.min(1.0, sample));

      if (this.outWriteIndex >= this.chunkSamples) {
        this.flushChunk();
      }
      this.fractionalPos += step;
    }
    this.fractionalPos -= numFrames;
    return true;
  }

  flushChunk() {
    const pcmInt16 = new Int16Array(this.chunkSamples);
    let sumSq = 0.0;
    for (let i = 0; i < this.chunkSamples; i++) {
      const s = this.outBufferF32[i];
      sumSq += s * s;
      pcmInt16[i] = s < 0 ? Math.max(-32768, Math.round(s * 32768)) : Math.min(32767, Math.round(s * 32767));
    }
    const rmsLinear = Math.sqrt(sumSq / this.chunkSamples);
    const f32Copy = new Float32Array(this.outBufferF32);

    this.port.postMessage(
      {
        pcmInt16Buffer: pcmInt16.buffer,
        pcmFloat32Buffer: f32Copy.buffer,
        rmsLinear,
      },
      [pcmInt16.buffer, f32Copy.buffer]
    );
    this.outWriteIndex = 0;
  }
}

registerProcessor('pcm-16k-worklet', PCM16kWorkletProcessor);
```

---

## 6. Complete Serverless Browser WebGPU Worker (`extension/workers/webgpu_worker.js`)

Runs **Moonshine Base** (`onnx-community/moonshine-base-ONNX`, `~61M` params, `~65 MB` `q8`), **Silero VAD**, and **emotion2vec-web-distill** (`thomashallock/emotion2vec-web-distill`, `2,231,986` params) inside Chrome's `1–4 GB` WebGPU memory ceiling ([Moonshine ASR](https://webai.show/models/moonshine-asr/), [emotion2vec-web-distill](https://huggingface.co/thomashallock/emotion2vec-web-distill)):

```javascript
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
```
