// Chrome Extension Content Script: Synchronizes Shadow DOM CwiRenderer overlay
// at 60-120 FPS to the active <video>.currentTime with zero CSS keyframe conflict.

let hostOverlay = null;
let shadowRoot = null;
let captionBoxEl = null;
let activeVideo = null;
let wordNodes = new Map();
let charNodes = new Map();
let wordSpecs = new Map();
let wordPhases = new Map();
let lockedWordIds = new Set();
let userIntensity = 1.0;
let rafRunning = false;

function smoothstep(edge0, edge1, x) {
  const t = Math.max(0.0, Math.min(1.0, (x - edge0) / Math.max(1e-5, edge1 - edge0)));
  return t * t * (3.0 - 2.0 * t);
}

function ensureVideoOverlay() {
  const videos = Array.from(document.querySelectorAll('video'));
  if (videos.length === 0) return null;

  activeVideo = videos.sort(
    (a, b) => b.clientWidth * b.clientHeight - a.clientWidth * a.clientHeight
  )[0];
  const parent = activeVideo.parentElement;
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

  if (!rafRunning) {
    rafRunning = true;
    requestAnimationFrame(tickOverlay);
  }
  return activeVideo;
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'RENDER_CAPTION_SPEC' && msg.payload) {
    ingestSpec(msg.payload);
  } else if (msg.type === 'APPLY_USER_PROFILE' && msg.profile) {
    userIntensity = Number(msg.profile.intensity ?? 1.0);
  }
});

function ingestSpec(spec) {
  const video = ensureVideoOverlay();
  if (!video || !captionBoxEl) return;

  const vhPx = video.clientHeight / 100.0;
  hostOverlay.style.setProperty('--cwi-vh', `${vhPx.toFixed(3)}px`);

  const k = userIntensity;
  for (const w of spec.words || []) {
    if (wordNodes.has(w.word_id)) {
      if (!lockedWordIds.has(w.word_id)) {
        const el = wordNodes.get(w.word_id);
        if (el.textContent !== w.text) {
          el.innerHTML = '';
          const chars = [];
          for (const ch of w.text) {
            const charSpan = document.createElement('span');
            charSpan.className = 'cwi-char';
            charSpan.textContent = ch;
            el.appendChild(charSpan);
            chars.push(charSpan);
          }
          charNodes.set(w.word_id, chars);
        }
        el.style.setProperty('--cwi-speaker-color', w.speaker.color_hex);
        wordSpecs.set(w.word_id, w);
      }
      continue;
    }

    const span = document.createElement('span');
    span.className = 'cwi-word';
    span.dataset.locale = spec.locale || 'en';

    const chars = [];
    for (const ch of w.text) {
      const charSpan = document.createElement('span');
      charSpan.className = 'cwi-char';
      charSpan.textContent = ch;
      charSpan.style.color = 'var(--cwi-readahead-color, #B0B5BC)';
      charSpan.style.transform = 'translate3d(0, 0em, 0) scale3d(1, 1, 1)';
      span.appendChild(charSpan);
      chars.push(charSpan);
    }
    charNodes.set(w.word_id, chars);

    span.style.setProperty('--cwi-speaker-color', w.speaker.color_hex);
    // Neutral font variation axes during initial read-ahead
    span.style.setProperty('--cwi-wght', '400');
    span.style.setProperty('--cwi-wdth', '100');
    span.style.setProperty('--cwi-grad', '0');

    captionBoxEl.appendChild(span);
    wordNodes.set(w.word_id, span);
    wordSpecs.set(w.word_id, w);
    wordPhases.set(w.word_id, 'unmounted');
  }
}

function tickOverlay() {
  if (activeVideo && captionBoxEl && wordSpecs.size > 0) {
    const currentMs = activeVideo.currentTime * 1000.0;
    const k = userIntensity;

    for (const [wordId, w] of wordSpecs.entries()) {
      const el = wordNodes.get(wordId);
      if (!el) continue;

      if (currentMs > w.end_ms + 4000) {
        el.remove();
        wordNodes.delete(wordId);
        charNodes.delete(wordId);
        wordSpecs.delete(wordId);
        wordPhases.delete(wordId);
        continue;
      }

      const chars = charNodes.get(wordId) || [];
      const prevPhase = wordPhases.get(wordId);

      // STATE 1: Read-Ahead Layer (currentMs < w.start_ms)
      if (currentMs < w.start_ms) {
        lockedWordIds.delete(wordId);
        if (prevPhase !== 'pre') {
          for (let i = 0; i < chars.length; i++) {
            chars[i].style.color = 'var(--cwi-readahead-color, #B0B5BC)';
            chars[i].style.transform = 'translate3d(0, 0em, 0) scale3d(1, 1, 1)';
          }
          el.style.setProperty('--cwi-wght', '400');
          el.style.setProperty('--cwi-wdth', '100');
          el.style.setProperty('--cwi-grad', '0');
          wordPhases.set(wordId, 'pre');
        }
        continue;
      }

      lockedWordIds.add(wordId);
      const t = w.typography;

      // STATE 3: Post-Onset / Completed Word (currentMs >= w.end_ms)
      if (currentMs >= w.end_ms) {
        if (prevPhase !== 'post') {
          for (let i = 0; i < chars.length; i++) {
            chars[i].style.color = 'var(--cwi-speaker-color, #4FC3F7)';
            chars[i].style.transform = 'translate3d(0, 0em, 0) scale3d(1, 1, 1)';
          }
          el.style.setProperty('--cwi-wght', `${Math.round(400 + k * (t.wght - 400))}`);
          el.style.setProperty('--cwi-wdth', `${Math.round(100 + k * (t.wdth - 100))}`);
          el.style.setProperty('--cwi-grad', `${Math.round(k * t.grad)}`);
          wordPhases.set(wordId, 'post');
        }
        continue;
      }

      // STATE 2: Active Word - Traveling Wave Packet across Characters (00:52 Synchronization)
      wordPhases.set(wordId, 'active');
      const dur = Math.max(80, w.duration_ms);
      const progress = Math.max(0.0, Math.min(1.0, (currentMs - w.start_ms) / dur));
      const numChars = chars.length;
      const sPeak = 1.0 + k * Math.max(0.15, (t.crest_scale * t.pop_scale - 1.0));
      const holdPeakEm = k * (t.hold_lift_em || 0.0);
      const sigma = Math.max(0.18, 1.2 / Math.max(1, numChars));

      // Lock typography parameters upon active onset
      el.style.setProperty('--cwi-wght', `${Math.round(400 + k * (t.wght - 400))}`);
      el.style.setProperty('--cwi-wdth', `${Math.round(100 + k * (t.wdth - 100))}`);
      el.style.setProperty('--cwi-grad', `${Math.round(k * t.grad)}`);

      // Detect if word is shouting (high vocal intensity / all-caps / exclamation / high crest scale)
      const cleanTxt = (w.text || '').replace(/[^a-zA-Z]/g, '');
      const isAllCaps = cleanTxt.length > 1 && cleanTxt === cleanTxt.toUpperCase();
      const hasExclamation = (w.text || '').endsWith('!');
      const isHighCrest = t && (t.crest_scale >= 1.30);
      const isHighProsody = w.prosody && (w.prosody.rms_db >= -18.0 && w.prosody.prominence_score >= 0.70);
      const isShout = isAllCaps || hasExclamation || isHighCrest || isHighProsody;

      if (isShout) {
        // SHOUT WORD: Entire word zooms in and zooms out synchronously according to timings (no wave format)
        const sShoutPeak = 1.0 + k * Math.max(0.25, (t.crest_scale * 1.20 - 1.0));
        const zoomEnvelope = Math.pow(Math.sin(progress * Math.PI), 0.75);
        const wordScale = 1.0 + (sShoutPeak - 1.0) * zoomEnvelope;
        const yLiftEm = holdPeakEm * Math.sin(progress * Math.PI);

        for (let i = 0; i < numChars; i++) {
          const charEl = chars[i];
          const ch = charEl.textContent || '';
          const isPunct = /^[.,\/#!$%\^&\*;:{}=\-_`~()]$/.test(ch);
          const charScale = isPunct ? 1.0 : wordScale;

          charEl.style.transform = `translate3d(0, ${yLiftEm.toFixed(4)}em, 0) scale3d(${charScale.toFixed(4)}, ${charScale.toFixed(4)}, 1)`;
          charEl.style.color = 'var(--cwi-speaker-color, #4FC3F7)';
        }
      } else {
        // NORMAL WORD: Traveling wave packet across characters (progressive phonetic wipe)
        for (let i = 0; i < numChars; i++) {
          const charEl = chars[i];
          const tau = (i + 0.5) / Math.max(1, numChars);
          const dist = tau - progress;
          const wave = Math.exp(-(dist * dist) / (2.0 * sigma * sigma));

          const ch = charEl.textContent || '';
          const isPunct = /^[.,\/#!$%\^&\*;:{}=\-_`~()]$/.test(ch);

          // Zero-latency attack: wave starts immediately with voice onset
          const charScale = isPunct ? 1.0 : (1.0 + (sPeak - 1.0) * wave);
          const yLiftEm = holdPeakEm * Math.sin(progress * Math.PI) * wave;

          charEl.style.transform = `translate3d(0, ${yLiftEm.toFixed(4)}em, 0) scale3d(${charScale.toFixed(4)}, ${charScale.toFixed(4)}, 1)`;

          // Horizontal color wipe strictly triggers as progress reaches character phonetic threshold
          // Character 0 turns speaker color instantly at progress >= 0.0 (exact voice onset)
          const charActivationThreshold = i / Math.max(1, numChars);
          if (progress >= charActivationThreshold) {
            charEl.style.color = 'var(--cwi-speaker-color, #4FC3F7)';
          } else {
            charEl.style.color = 'var(--cwi-readahead-color, #B0B5BC)';
          }
        }
      }
    }
  }
  requestAnimationFrame(tickOverlay);
}
