# Standalone React 19 Video Player, Voice Compass Ring & Extension Content Script (`react_player_and_popup.md`)

> **Primary References:** [Expressive Subtitle System Architecture Blueprint (Section 5.3 & 7)](https://docs.google.com/document/d/1z1Y8Y7amQr7zsJ3YKHucA1Q5DflC_bMg_6MuSrtAIcc/edit?tab=t.0), [Weave Voice Compass (SNU, 2026)](https://xlaude2040.com/weave/), [CwiRenderer Customization Profiles](https://stardance.hackclub.com/projects/63123), [STEM Adaptive Captioning Profiles](https://arxiv.org/pdf/2603.15977)

---

## 1. Chrome Extension Content Script — `extension/content/content.js`

Mounts the Shadow DOM `CwiRenderer` overlay directly over the largest active `<video>` element on the page (YouTube, Twitch, Vimeo, Google Meet, etc.):

```javascript
// Content script injected into web pages: locates the active <video>, attaches an isolated
// Shadow DOM overlay, loads overlay.css, and reconciles incoming CaptionSpec v2.0-cwi packets.

let hostOverlay = null;
let shadowRoot = null;
let captionBoxEl = null;
let wordNodes = new Map();
let lockedWordIds = new Set();
let userIntensity = 1.0;

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
  }
});

function renderCaptionSpecOnVideo(spec) {
  const video = ensureVideoOverlay();
  if (!video || !captionBoxEl) return;

  const vhPx = video.clientHeight / 100;
  hostOverlay.style.setProperty('--cwi-vh', `${vhPx}px`);

  const acousticMs = spec.stream_acoustic_time_ms ?? Math.round(video.currentTime * 1000) + 1750;
  const presentMs = acousticMs - (spec.presentation_offset_ms || 1750);

  for (const w of spec.words || []) {
    if (presentMs >= w.start_ms) {
      lockedWordIds.add(w.word_id);
    }

    const existing = wordNodes.get(w.word_id);
    if (existing) {
      if (lockedWordIds.has(w.word_id)) continue;
      existing.textContent = w.text;
      existing.style.setProperty('--cwi-speaker-color', w.speaker.color_hex);
      continue;
    }

    const span = document.createElement('span');
    span.className = 'cwi-word';
    span.dataset.locale = spec.locale || 'en';
    span.textContent = w.text;

    const delayMs = Math.max(420, w.start_ms - presentMs);
    const k = userIntensity;
    const t = w.typography;

    span.style.setProperty('--cwi-onset-delay-ms', `${delayMs}ms`);
    span.style.setProperty('--cwi-duration-ms', `${w.duration_ms}ms`);
    span.style.setProperty('--cwi-speaker-color', w.speaker.color_hex);
    span.style.setProperty('--cwi-wght', `${Math.round(400 + k * (t.wght - 400))}`);
    span.style.setProperty('--cwi-wdth', `${Math.round(100 + k * (t.wdth - 100))}`);
    span.style.setProperty('--cwi-grad', `${Math.round(k * t.grad)}`);
    span.style.setProperty('--cwi-pop-scale', `${(1.0 + k * (t.pop_scale - 1.0)).toFixed(3)}`);
    span.style.setProperty('--cwi-crest-peak', `${(1.0 + k * (t.crest_scale - 1.0)).toFixed(4)}`);
    span.style.setProperty('--cwi-crest-sustain', `${(1.0 + k * (t.crest_sustain_scale - 1.0)).toFixed(4)}`);
    span.style.setProperty('--cwi-hold-em', `${(k * t.hold_lift_em).toFixed(4)}em`);

    captionBoxEl.appendChild(span);
    wordNodes.set(w.word_id, span);
  }
}
```

---

## 2. SVG Voice Compass Ring Renderer — `frontend_renderer/src/core/voice_compass.ts`

Renders the 360° directional speaker bearing arc when paired with the wearable ReSpeaker XVF3800 microphone array ([Weave](https://xlaude2040.com/weave/)):

```typescript
export function computeCompassArcPath(
  cx: number,
  cy: number,
  radius: number,
  bearingDeg: number,
  spreadDeg: number
): string {
  const halfSpread = Math.max(8.0, Math.min(90.0, spreadDeg / 2.0));
  // Convert azimuth (-180..+180 where 0 is top/12-o'clock) to SVG polar radians
  const startAngleRad = ((bearingDeg - halfSpread - 90.0) * Math.PI) / 180.0;
  const endAngleRad = ((bearingDeg + halfSpread - 90.0) * Math.PI) / 180.0;

  const x1 = cx + radius * Math.cos(startAngleRad);
  const y1 = cy + radius * Math.sin(startAngleRad);
  const x2 = cx + radius * Math.cos(endAngleRad);
  const y2 = cy + radius * Math.sin(endAngleRad);

  const largeArcFlag = halfSpread * 2.0 > 180.0 ? 1 : 0;
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}
```
