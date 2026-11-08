// Offscreen Audio Capture + Mandatory Speaker Passthrough + Auto-Reconnecting WebSocket
// + 20s MV3 Service Worker Keepalive Heartbeat.

let audioCtx = null;
let mediaStream = null;
let workletNode = null;
let wsClient = null;
let keepaliveTimer = null;
let reconnectTimer = null;
let isCapturing = false;
let currentLocale = 'en';

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'OFFSCREEN_INIT_CAPTURE') {
    startCapture(msg.streamId, msg.mode, msg.locale);
  } else if (msg.type === 'OFFSCREEN_STOP_CAPTURE') {
    stopCapture();
  }
});

async function startCapture(streamId, mode = 'server_ws', locale = 'en') {
  stopCapture();
  isCapturing = true;
  currentLocale = locale;

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

  // MANDATORY Speaker Passthrough so chrome.tabCapture never mutes user audio
  sourceNode.connect(audioCtx.destination);

  await audioCtx.audioWorklet.addModule('pcm-worklet.js');
  workletNode = new AudioWorkletNode(audioCtx, 'pcm-16k-worklet');
  sourceNode.connect(workletNode);

  // Send 20s heartbeat to background.js so Chrome MV3 service worker never idles out
  keepaliveTimer = setInterval(() => {
    chrome.runtime.sendMessage({ type: 'OFFSCREEN_HEARTBEAT', ts: Date.now() }).catch(() => {});
  }, 20000);

  connectWebSocket(workletNode.port, 0);
}

function connectWebSocket(workletPort, retryCount = 0) {
  if (!isCapturing) return;
  wsClient = new WebSocket('ws://127.0.0.1:8765/ws');
  wsClient.binaryType = 'arraybuffer';

  wsClient.onopen = () => {
    retryCount = 0;
    wsClient.send(
      JSON.stringify({
        type: 'SESSION_START',
        session_id: `sess_${Date.now()}`,
        sample_rate: 16000,
        channels: 1,
        format: 'pcm_s16le',
        mode: 'live_streaming',
        locale: currentLocale,
      })
    );
  };

  wsClient.onmessage = (event) => {
    if (typeof event.data === 'string') {
      try {
        const spec = JSON.parse(event.data);
        if (spec.schema_version === '2.0-cwi') {
          chrome.runtime.sendMessage({
            type: 'CAPTION_SPEC_PAYLOAD',
            payload: spec,
          });
        }
      } catch (_) {}
    }
  };

  wsClient.onclose = () => {
    if (isCapturing) {
      const delayMs = Math.min(5000, 500 * Math.pow(1.5, retryCount));
      reconnectTimer = setTimeout(() => connectWebSocket(workletPort, retryCount + 1), delayMs);
    }
  };

  workletPort.onmessage = (e) => {
    if (wsClient && wsClient.readyState === WebSocket.OPEN) {
      wsClient.send(e.data.pcmInt16Buffer);
    }
  };
}

function stopCapture() {
  isCapturing = false;
  if (keepaliveTimer) {
    clearInterval(keepaliveTimer);
    keepaliveTimer = null;
  }
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
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
}
