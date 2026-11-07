// Content script injected into web pages: locates the active <video>, attaches an isolated
// Shadow DOM overlay, loads overlay.css, and reconciles incoming CaptionSpec v2.0-cwi packets.

let hostOverlay = null;
let shadowRoot = null;
let captionBoxEl = null;
let wordNodes = new Map();
let wordEndTimes = new Map();
let lockedWordIds = new Set();
let userIntensity = 1.0;
let userProfile = 'default_cwi';

const isSafari =
  typeof navigator !== 'undefined' &&
  /^((?!chrome|android).)*safari/i.test(navigator.userAgent);

// Load persisted settings on load if available
if (typeof chrome !== 'undefined' && chrome.storage?.local) {
  chrome.storage.local.get(['cwi_intensity', 'cwi_profile'], (res) => {
    if (res) {
      if (res.cwi_intensity !== undefined) {
        userIntensity = Number(res.cwi_intensity) / 100.0;
      }
      if (res.cwi_profile) {
        userProfile = res.cwi_profile;
      }
    }
  });
}

function ensureVideoOverlay() {
  const videos = Array.from(document.querySelectorAll('video'));
  if (videos.length === 0) return null;

  // Pick largest visible video element
  const video = videos.sort(
    (a, b) => b.clientWidth * b.clientHeight - a.clientWidth * a.clientHeight
  )[0];
  const parent = video.parentElement;
  if (!parent) return null;

  if (getComputedStyle(parent).position === 'static') {
    parent.style.position = 'relative';
  }

  if (!hostOverlay || !parent.contains(hostOverlay)) {
    hostOverlay = document.createElement('div');
    hostOverlay.id = 'cwi-shadow-host';
    hostOverlay.style.cssText = 'position:absolute;inset:0;pointer-events:none;z-index:2147483647;';
    parent.appendChild(hostOverlay);

    shadowRoot = hostOverlay.attachShadow({ mode: 'open' });
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = chrome.runtime.getURL('content/overlay.css');
    shadowRoot.appendChild(link);

    const workArea = document.createElement('div');
    workArea.className = 'cwi-work-area';
    captionBoxEl = document.createElement('div');
    captionBoxEl.className = 'cwi-caption-box';
    workArea.appendChild(captionBoxEl);
    shadowRoot.appendChild(workArea);
  }
  return video;
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'RENDER_CAPTION_SPEC' && msg.payload) {
    renderCaptionSpecOnVideo(msg.payload);
  } else if (msg.type === 'APPLY_USER_PROFILE' && msg.profile) {
    userIntensity = Number(msg.profile.intensity ?? 1.0);
    userProfile = msg.profile.name || 'default_cwi';
  }
});

function renderCaptionSpecOnVideo(spec) {
  const video = ensureVideoOverlay();
  if (!video || !captionBoxEl) return;

  const vhPx = video.clientHeight / 100;
  hostOverlay.style.setProperty('--cwi-vh', `${vhPx}px`);

  const acousticMs = spec.stream_acoustic_time_ms ?? Math.round(video.currentTime * 1000) + 1750;
  const presentMs = acousticMs - (spec.presentation_offset_ms || 1750);

  // Render non-speech / paralinguistic audio events (<|BGM|>, [applause], [laughter], etc.)
  if (spec.non_speech_events && Array.isArray(spec.non_speech_events)) {
    for (const nse of spec.non_speech_events) {
      const nseId = `nse_${nse.start_ms}_${nse.label}`;
      if (!wordNodes.has(nseId)) {
        const badge = document.createElement('span');
        badge.className = 'cwi-sfx-badge';
        badge.textContent = nse.label;
        captionBoxEl.appendChild(badge);
        wordNodes.set(nseId, badge);
        wordEndTimes.set(nseId, nse.end_ms);
      }
    }
  }

  // Reconcile words
  for (const w of spec.words || []) {
    // IMMUTABLE POST-ONSET TEXT RULE:
    // Once presentation clock crosses word start_ms, text & layout are locked.
    if (presentMs >= w.start_ms) {
      lockedWordIds.add(w.word_id);
    }

    const existing = wordNodes.get(w.word_id);
    if (existing) {
      if (lockedWordIds.has(w.word_id)) {
        continue;
      }
      existing.textContent = w.text;
      if (w.speaker?.color_hex) {
        existing.style.setProperty('--cwi-speaker-color', w.speaker.color_hex);
      }
      continue;
    }

    // New word node created in read-ahead zone
    const span = document.createElement('span');
    span.className = 'cwi-word';
    span.dataset.wordId = w.word_id;
    span.dataset.locale = spec.locale || 'en';
    span.textContent = w.text;

    const delayMs = Math.max(420, w.start_ms - presentMs);
    const k = Math.max(0.0, Math.min(1.0, userIntensity));
    const t = w.typography || {
      wght: 400,
      wdth: 100,
      grad: 0,
      pop_scale: 1.15,
      crest_scale: 1.0,
      crest_sustain_scale: 1.0,
      hold_lift_em: 0.0,
    };

    let wght = Math.round(400 + k * (t.wght - 400));
    // Safari WebKit wght:400 bug workaround
    if (isSafari && wght >= 398 && wght <= 402) {
      wght = 404;
    }

    const wdth = Math.round(100 + k * (t.wdth - 100));
    const grad = Math.round(k * t.grad);
    const popScale = (1.0 + k * ((t.pop_scale ?? 1.15) - 1.0)).toFixed(3);
    const crestPeak = (1.0 + k * ((t.crest_scale ?? 1.0) - 1.0)).toFixed(4);
    const crestSustain = (1.0 + k * ((t.crest_sustain_scale ?? 1.0) - 1.0)).toFixed(4);
    const holdEm = `${(k * (t.hold_lift_em ?? 0.0)).toFixed(4)}em`;

    span.style.setProperty('--cwi-onset-delay-ms', `${delayMs}ms`);
    span.style.setProperty('--cwi-duration-ms', `${w.duration_ms}ms`);
    span.style.setProperty('--cwi-speaker-color', w.speaker?.color_hex || '#4FC3F7');
    span.style.setProperty('--cwi-wght', `${wght}`);
    span.style.setProperty('--cwi-wdth', `${wdth}`);
    span.style.setProperty('--cwi-grad', `${grad}`);
    span.style.setProperty('--cwi-pop-scale', popScale);
    span.style.setProperty('--cwi-crest-peak', crestPeak);
    span.style.setProperty('--cwi-crest-sustain', crestSustain);
    span.style.setProperty('--cwi-hold-em', holdEm);

    captionBoxEl.appendChild(span);
    wordNodes.set(w.word_id, span);
    wordEndTimes.set(w.word_id, w.end_ms);
  }

  // Periodic pruning of expired words (> 4500ms after end_ms) to keep DOM lean
  for (const [id, endMs] of wordEndTimes.entries()) {
    if (presentMs > endMs + 4500) {
      const el = wordNodes.get(id);
      if (el) el.remove();
      wordNodes.delete(id);
      wordEndTimes.delete(id);
      lockedWordIds.delete(id);
    }
  }
}
