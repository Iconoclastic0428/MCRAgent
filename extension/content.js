(function () {
  const ADVISOR_BASE = 'http://127.0.0.1:8765';

  function injectObserver() {
    const script = document.createElement('script');
    script.src = chrome.runtime.getURL('page-observer.js');
    script.onload = () => script.remove();
    (document.documentElement || document.head).appendChild(script);
  }

  function createOverlay() {
    if (document.getElementById('tziakcha-mcr-advisor-overlay')) return;
    const overlay = document.createElement('div');
    overlay.id = 'tziakcha-mcr-advisor-overlay';
    overlay.style.cssText = [
      'position:fixed',
      'right:16px',
      'top:16px',
      'z-index:2147483647',
      'min-width:220px',
      'max-width:360px',
      'padding:12px 14px',
      'border:1px solid rgba(125,211,252,.7)',
      'border-radius:8px',
      'background:rgba(12,18,24,.9)',
      'color:#e8edf2',
      'font:14px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif',
      'box-shadow:0 10px 30px rgba(0,0,0,.35)',
      'pointer-events:none'
    ].join(';');
    overlay.innerHTML = [
      '<div style="font-size:11px;color:#9fb0c0;text-transform:uppercase;letter-spacing:.04em">MCR Model Advisor</div>',
      '<div data-role="rec" style="font-size:22px;font-weight:750;color:#7dd3fc;margin-top:3px">Waiting</div>',
      '<div data-role="hand" style="font-size:12px;color:#c6d0dc;margin-top:5px;line-height:1.35"></div>'
    ].join('');
    (document.body || document.documentElement).appendChild(overlay);
  }

  async function postObservedPayload(payload) {
    await fetch(`${ADVISOR_BASE}/observe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  }

  async function refreshOverlay() {
    const overlay = document.getElementById('tziakcha-mcr-advisor-overlay');
    if (!overlay) return;
    try {
      const [state, rec] = await Promise.all([
        fetch(`${ADVISOR_BASE}/state`).then((response) => response.json()),
        fetch(`${ADVISOR_BASE}/recommendation`).then((response) => response.json())
      ]);
      overlay.querySelector('[data-role="rec"]').textContent = rec.text || 'Waiting';
      overlay.querySelector('[data-role="hand"]').textContent = (state.hand_display || []).join(' ');
    } catch (error) {
      overlay.querySelector('[data-role="rec"]').textContent = 'Advisor offline';
      overlay.querySelector('[data-role="hand"]').textContent = '';
    }
  }

  injectObserver();

  window.addEventListener('message', async (event) => {
    if (event.source !== window) return;
    const message = event.data;
    if (!message || message.source !== 'tziakcha-mcr-observer') return;
    try {
      await postObservedPayload(message.payload);
    } catch (error) {
      console.debug('[Tziakcha MCR Observer] local advisor unavailable', error);
    }
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createOverlay, { once: true });
  } else {
    createOverlay();
  }
  setInterval(refreshOverlay, 1000);
})();
