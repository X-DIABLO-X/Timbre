// Chrome Extension Content Script: Synchronizes Shadow DOM CwiRenderer overlay
// at 60-120 FPS to the active <video>.currentTime with zero CSS keyframe conflict.

let hostOverlay = null;
let shadowRoot = null;
let captionBoxEl = null;
let activeVideo = null;
let wordNodes = new Map();
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
        el.textContent = w.text;
        el.style.setProperty('--cwi-speaker-color', w.speaker.color_hex);
        wordSpecs.set(w.word_id, w);
      }
      continue;
    }

    const span = document.createElement('span');
    span.className = 'cwi-word';
    span.dataset.locale = spec.locale || 'en';
    span.textContent = w.text;

    const t = w.typography;
    span.style.setProperty('--cwi-speaker-color', w.speaker.color_hex);
    span.style.setProperty('--cwi-wght', `${Math.round(400 + k * (t.wght - 400))}`);
    span.style.setProperty('--cwi-wdth', `${Math.round(100 + k * (t.wdth - 100))}`);
    span.style.setProperty('--cwi-grad', `${Math.round(k * t.grad)}`);

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
        wordSpecs.delete(wordId);
        wordPhases.delete(wordId);
        continue;
      }

      const prevPhase = wordPhases.get(wordId);
      if (currentMs < w.start_ms) {
        if (prevPhase !== 'pre') {
          el.style.backgroundPosition = '100% 0%';
          el.style.transform = 'translate3d(0, 0em, 0) scale3d(1, 1, 1)';
          wordPhases.set(wordId, 'pre');
        }
        continue;
      }

      lockedWordIds.add(wordId);
      const t = w.typography;
      // Fixed: Words never shrink below 1.0 unprompted; normal speech rests at 1.0x scale
      const crestScaleEff = Math.max(1.0, 1.0 + k * (t.crest_scale - 1.0));
      const sSustain = crestScaleEff;

      if (currentMs >= w.end_ms) {
        if (prevPhase !== 'post') {
          el.style.backgroundPosition = '0% 0%';
          el.style.transform = 'translate3d(0, 0em, 0) scale3d(1, 1, 1)';
          wordPhases.set(wordId, 'post');
        }
        continue;
      }

      wordPhases.set(wordId, 'active');
      const dur = Math.max(80, w.duration_ms);
      const progress = Math.max(0.0, Math.min(1.0, (currentMs - w.start_ms) / dur));
      el.style.backgroundPosition = `${((1.0 - progress) * 100.0).toFixed(2)}% 0%`;

      const sPeak = (1.0 + k * (t.pop_scale - 1.0)) * crestScaleEff;
      let scaleNow;
      if (progress <= 0.22) {
        scaleNow = 1.0 + smoothstep(0.0, 0.22, progress) * (sPeak - 1.0);
      } else if (progress <= 0.55) {
        scaleNow = sPeak + smoothstep(0.22, 0.55, progress) * (sSustain - sPeak);
      } else {
        scaleNow = sSustain;
      }
      const yLiftEm = k * t.hold_lift_em * Math.sin(progress * Math.PI);
      el.style.transform = `translate3d(0, ${yLiftEm.toFixed(4)}em, 0) scale3d(${scaleNow.toFixed(4)}, ${scaleNow.toFixed(4)}, 1)`;
    }
  }
  requestAnimationFrame(tickOverlay);
}
