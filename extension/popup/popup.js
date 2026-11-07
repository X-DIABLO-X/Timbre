// Popup controller: manages user preferences, toggles capture sessions,
// and sends realtime profile updates to the background worker and content script.

document.addEventListener('DOMContentLoaded', async () => {
  const btnToggle = document.getElementById('btn-toggle-session');
  const statusBadge = document.getElementById('status-badge');
  const statusText = document.getElementById('status-text');
  const statusMsg = document.getElementById('status-msg');
  const modeServer = document.getElementById('mode-server');
  const modeWebgpu = document.getElementById('mode-webgpu');
  const selectProfile = document.getElementById('select-profile');
  const sliderIntensity = document.getElementById('slider-intensity');
  const intensityVal = document.getElementById('intensity-val');
  const selectLocale = document.getElementById('select-locale');

  const isSafari =
    typeof navigator !== 'undefined' &&
    /^((?!chrome|android).)*safari/i.test(navigator.userAgent);

  let isSessionActive = false;

  // 1. Load persisted preferences from chrome.storage.local
  chrome.storage.local.get(
    ['cwi_mode', 'cwi_profile', 'cwi_intensity', 'cwi_locale'],
    (stored) => {
      if (stored.cwi_mode === 'browser_webgpu') {
        modeWebgpu.checked = true;
      } else {
        modeServer.checked = true;
      }

      if (stored.cwi_profile) {
        selectProfile.value = stored.cwi_profile;
      }

      if (stored.cwi_intensity !== undefined) {
        sliderIntensity.value = stored.cwi_intensity;
        intensityVal.textContent = `${stored.cwi_intensity}%`;
      }

      if (stored.cwi_locale) {
        selectLocale.value = stored.cwi_locale;
      }
    }
  );

  // 2. Query background service worker for current session status
  try {
    const res = await chrome.runtime.sendMessage({ type: 'GET_CWI_STATUS' });
    if (res && res.active) {
      setSessionActiveUI(true);
    } else {
      setSessionActiveUI(false);
    }
  } catch (e) {
    setSessionActiveUI(false);
  }

  // Helper to update active/inactive UI
  function setSessionActiveUI(active) {
    isSessionActive = active;
    if (active) {
      statusBadge.classList.add('active');
      statusText.textContent = 'Active Capturing';
      btnToggle.className = 'btn-action btn-stop';
      btnToggle.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
          <rect x="6" y="6" width="12" height="12"></rect>
        </svg>
        <span>Stop Captions</span>
      `;
      statusMsg.textContent = 'Streaming 16kHz audio to CWI Engine...';
      statusMsg.className = 'status-msg';
    } else {
      statusBadge.classList.remove('active');
      statusText.textContent = 'Idle';
      btnToggle.className = 'btn-action btn-start';
      btnToggle.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
        <span>Start Expressive Captions</span>
      `;
      statusMsg.textContent = 'Select options and click Start to begin.';
      statusMsg.className = 'status-msg';
    }
  }

  function broadcastProfileUpdate() {
    const intensityFloat = Number(sliderIntensity.value) / 100.0;
    const profileName = selectProfile.value;
    const profilePayload = {
      name: profileName,
      intensity: intensityFloat,
      isSafari,
    };

    chrome.storage.local.set({
      cwi_profile: profileName,
      cwi_intensity: Number(sliderIntensity.value),
    });

    chrome.runtime.sendMessage({
      type: 'UPDATE_USER_PROFILE',
      profile: profilePayload,
    }).catch(() => {});
  }

  // 3. Event Listeners

  // Intensity slider
  sliderIntensity.addEventListener('input', () => {
    intensityVal.textContent = `${sliderIntensity.value}%`;
    broadcastProfileUpdate();
  });

  // Profile select
  selectProfile.addEventListener('change', () => {
    if (selectProfile.value === 'minimal_static') {
      sliderIntensity.value = 0;
      intensityVal.textContent = '0%';
    } else if (Number(sliderIntensity.value) === 0) {
      sliderIntensity.value = 100;
      intensityVal.textContent = '100%';
    }
    broadcastProfileUpdate();
  });

  // Mode radio change
  const handleModeChange = () => {
    const chosenMode = modeWebgpu.checked ? 'browser_webgpu' : 'server_ws';
    chrome.storage.local.set({ cwi_mode: chosenMode });
  };
  modeServer.addEventListener('change', handleModeChange);
  modeWebgpu.addEventListener('change', handleModeChange);

  // Locale select
  selectLocale.addEventListener('change', () => {
    chrome.storage.local.set({ cwi_locale: selectLocale.value });
  });

  // Start / Stop Toggle
  btnToggle.addEventListener('click', async () => {
    statusMsg.className = 'status-msg';
    statusMsg.textContent = '';

    if (isSessionActive) {
      // Stop session
      try {
        await chrome.runtime.sendMessage({ type: 'STOP_CWI_SESSION' });
        setSessionActiveUI(false);
      } catch (err) {
        statusMsg.className = 'status-msg error';
        statusMsg.textContent = `Error stopping session: ${err.message}`;
      }
      return;
    }

    // Start session
    try {
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tabs || tabs.length === 0) {
        statusMsg.className = 'status-msg error';
        statusMsg.textContent = 'No active browser tab found.';
        return;
      }

      const activeTab = tabs[0];
      if (activeTab.url && (activeTab.url.startsWith('chrome://') || activeTab.url.startsWith('edge://') || activeTab.url.startsWith('about:'))) {
        statusMsg.className = 'status-msg error';
        statusMsg.textContent = 'Cannot capture internal browser pages. Please open YouTube, Twitch, or a video page.';
        return;
      }

      const mode = modeWebgpu.checked ? 'browser_webgpu' : 'server_ws';
      const locale = selectLocale.value;

      statusMsg.textContent = mode === 'browser_webgpu'
        ? 'Initializing WebGPU Worker & Tab Capture...'
        : 'Connecting Native Host & Tab Capture...';

      const response = await chrome.runtime.sendMessage({
        type: 'START_CWI_SESSION',
        tabId: activeTab.id,
        mode,
        locale,
      });

      if (response && response.ok) {
        setSessionActiveUI(true);
      } else {
        throw new Error(response?.error || 'Failed to initialize session');
      }
    } catch (err) {
      statusMsg.className = 'status-msg error';
      statusMsg.textContent = `Start failed: ${err.message}`;
      setSessionActiveUI(false);
    }
  });
});
