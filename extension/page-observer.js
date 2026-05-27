(function () {
  if (window.__tziakchaMcrObserverInstalled) return;
  window.__tziakchaMcrObserverInstalled = true;

  const NativeWebSocket = window.WebSocket;

  function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let index = 0; index < bytes.byteLength; index += 1) {
      binary += String.fromCharCode(bytes[index]);
    }
    return btoa(binary);
  }

  function mirror(data) {
    let payload = null;
    if (typeof data === 'string') {
      payload = { kind: 'text', data };
    } else if (data instanceof ArrayBuffer) {
      payload = { kind: 'binary', base64: arrayBufferToBase64(data) };
    } else if (ArrayBuffer.isView(data)) {
      const viewBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
      payload = { kind: 'binary', base64: arrayBufferToBase64(viewBuffer) };
    }
    if (payload) {
      window.postMessage({ source: 'tziakcha-mcr-observer', payload }, '*');
    }
  }

  function ObservedWebSocket(url, protocols) {
    const socket = protocols === undefined
      ? new NativeWebSocket(url)
      : new NativeWebSocket(url, protocols);
    socket.addEventListener('message', (event) => mirror(event.data));
    return socket;
  }

  ObservedWebSocket.prototype = NativeWebSocket.prototype;
  ObservedWebSocket.CONNECTING = NativeWebSocket.CONNECTING;
  ObservedWebSocket.OPEN = NativeWebSocket.OPEN;
  ObservedWebSocket.CLOSING = NativeWebSocket.CLOSING;
  ObservedWebSocket.CLOSED = NativeWebSocket.CLOSED;

  window.WebSocket = ObservedWebSocket;
})();
