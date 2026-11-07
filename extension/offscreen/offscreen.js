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
        }).catch(() => {});
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
      }).catch(() => {});
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
