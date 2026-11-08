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

  if (message.type === 'GET_CWI_STATUS') {
    sendResponse({ active: activeTabId !== null, activeTabId });
    return false;
  }

  // Acknowledge offscreen heartbeat to keep MV3 service worker active
  if (message.type === 'OFFSCREEN_HEARTBEAT') {
    sendResponse({ ok: true, ts: Date.now() });
    return false;
  }


  // Forward CaptionSpec v2.0-cwi emitted by offscreen.js (via WS or WebGPU Worker) to active tab content.js
  if (message.type === 'CAPTION_SPEC_PAYLOAD' && activeTabId !== null) {
    chrome.tabs.sendMessage(activeTabId, {
      type: 'RENDER_CAPTION_SPEC',
      payload: message.payload,
    }).catch((err) => {
      console.debug('[CWI] Failed to forward CaptionSpec to content script:', err);
    });
  }

  // Forward user customization updates (intensity slider 0-100%, exception profile)
  if (message.type === 'UPDATE_USER_PROFILE' && activeTabId !== null) {
    chrome.tabs.sendMessage(activeTabId, {
      type: 'APPLY_USER_PROFILE',
      profile: message.profile,
    }).catch((err) => {
      console.debug('[CWI] Failed to forward user profile to content script:', err);
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
  await chrome.runtime.sendMessage({ type: 'OFFSCREEN_STOP_CAPTURE' }).catch(() => {});
  activeTabId = null;
}
