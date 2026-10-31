// Sentence streaming shares the workshop's Generate action and finished player.
// The request owns its reader, AudioContext and scheduled sources. Cancellation
// closes all three; a late decode or network packet can never revive old speech.
export function createSpeechStream() {
 let active: { controller: AbortController; context: AudioContext; sources: Set<AudioBufferSourceNode> } | undefined;
 const stop = document.createElement('button');
 stop.type = 'button'; stop.id = 'stream-stop-btn'; stop.className = 'btn secondary';
 stop.textContent = 'Stop speech'; stop.hidden = true;
 // Keep Stop above the existing blocking generation overlay.
 Object.assign(stop.style, { position: 'fixed', bottom: '24px', right: '24px', zIndex: '10000' });
 document.body.append(stop);
 const status = document.createElement('p');
 status.id = 'stream-status'; status.setAttribute('role', 'status');
 document.querySelector('#generate-btn')?.parentElement?.after(status);
 function cancel() {
  const turn = active;
  if (!turn) return;
  active = undefined; turn.controller.abort();
  for (const source of turn.sources) { try { source.stop(); } catch {} }
  void turn.context.close(); stop.hidden = true;
 }
 stop.addEventListener('click', cancel);
 window.addEventListener('pagehide', cancel);
 document.addEventListener('visibilitychange', () => { if (document.hidden) cancel(); });
 document.querySelectorAll('input[name="voice_mode"]').forEach(control => control.addEventListener('change', cancel));
 // Changing the input or voice invalidates speech belonging to the earlier turn.
 for (const id of ['text', 'predefined-voice-select', 'clone-reference-select', 'output-format', 'voice-mode']) {
 document.getElementById(id)?.addEventListener(id === 'text' ? 'input' : 'change', cancel);
 }
 return {
  cancel,
  async generate(request: unknown) {
   cancel();
   const turn = { controller: new AbortController(), context: new AudioContext({ sampleRate: 24000 }), sources: new Set<AudioBufferSourceNode>() };
   active = turn; stop.hidden = false;
   status.textContent = 'Generating speech…';
   const check = () => { if (active !== turn || turn.controller.signal.aborted) throw new DOMException('Speech interrupted', 'AbortError'); };
   let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
   let complete: any, samples = 0, frames = 0, nextAt = 0, played = false, transportFinished = false;
   const pcm: Uint8Array[] = [];
   const started = performance.now();
   try {
    // Resume inside the user's Generate gesture; autoplay policy is never bypassed.
    await turn.context.resume(); check();
    const response = await fetch('/tts/stream', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request), signal: turn.controller.signal });
    if (!response.ok) throw new Error((await response.json()).detail || 'Speech request failed');
    if (!response.headers.get('content-type')?.startsWith('application/x-ndjson') || !response.body) throw new Error('Invalid speech stream');
    reader = response.body.getReader();
    let pending = '', began = false;
    const decoder = new TextDecoder('utf-8', { fatal: true });
    async function frame(item: any) {
     check();
     if (complete) throw new Error('Speech data after completion');
     if (item.type === 'start') {
      if (began || item.version !== 1 || item.sample_rate !== 24000 || item.format !== 'wav') throw new Error('Unsupported speech stream');
      began = true; return;
     }
     if (!began) throw new Error('Speech stream has no start');
     if (item.type === 'error') throw new Error(item.detail || 'Speech generation failed');
     if (item.type === 'complete') {
      if (!frames || item.frames !== frames || item.samples !== samples) throw new Error('Incomplete speech stream');
      complete = item; return;
     }
     if (item.type !== 'audio' || item.index !== frames || item.offset_samples !== samples) throw new Error('Out-of-order speech frame');
     const raw = Uint8Array.from(atob(item.wav), c => c.charCodeAt(0));
     const view = new DataView(raw.buffer);
     if (raw.length < 46 || String.fromCharCode(...raw.subarray(0, 4)) !== 'RIFF' ||
         String.fromCharCode(...raw.subarray(8, 16)) !== 'WAVEfmt ' ||
         String.fromCharCode(...raw.subarray(36, 40)) !== 'data' ||
         view.getUint32(4, true) !== raw.length - 8 || view.getUint32(16, true) !== 16 ||
         view.getUint16(20, true) !== 1 || view.getUint16(22, true) !== 1 ||
         view.getUint32(24, true) !== 24000 || view.getUint32(28, true) !== 48000 ||
         view.getUint16(32, true) !== 2 || view.getUint16(34, true) !== 16 ||
         view.getUint32(40, true) !== raw.length - 44 || (raw.length - 44) % 2 ||
         item.samples !== (raw.length - 44) / 2) throw new Error('Malformed speech audio');
     // Bound scheduled audio. Pausing network reads propagates backpressure to
     // the service's one-frame channel rather than queuing the whole narration.
     while (nextAt - turn.context.currentTime > 5) {
      await new Promise<void>((resolve, reject) => {
       const timer = setTimeout(done, 25);
       function done() { turn.controller.signal.removeEventListener('abort', aborted); resolve(); }
       function aborted() { clearTimeout(timer); reject(new DOMException('Speech interrupted', 'AbortError')); }
       turn.controller.signal.addEventListener('abort', aborted, { once: true });
      }); check();
     }
     const buffer = turn.context.createBuffer(1, item.samples, 24000), channel = buffer.getChannelData(0);
     for (let n = 0; n < item.samples; n++) channel[n] = view.getInt16(44 + n * 2, true) / 32768;
     check();
     const source = turn.context.createBufferSource(); source.buffer = buffer; source.connect(turn.context.destination);
     source.onended = () => { turn.sources.delete(source); source.disconnect(); if (transportFinished && !turn.sources.size && active === turn) { active = undefined; stop.hidden = true; void turn.context.close(); } };
     turn.sources.add(source);
     const gap = played ? Math.max(0, turn.context.currentTime - nextAt) : 0;
     nextAt = Math.max(nextAt, turn.context.currentTime + (played && gap === 0 ? 0 : .03));
     source.start(nextAt); nextAt += buffer.duration; played = true;
     frames++; samples += item.samples; pcm.push(raw.slice(44));
     status.textContent = 'Speaking · ' + frames + ' segment' + (frames === 1 ? '' : 's');
     // Events report actual scheduled playback, including measured starvation.
     status.dispatchEvent(new CustomEvent('myth-stream-frame', { bubbles: true, detail: { index: item.index, samples: item.samples, gap_seconds: gap, received_ms: performance.now() - started, scheduled_end: nextAt, current_time: turn.context.currentTime } }));
    }
    while (true) {
     const result = await reader.read(); check();
     pending += decoder.decode(result.value, { stream: !result.done });
     let split;
     while ((split = pending.indexOf('\n')) >= 0) {
      const line = pending.slice(0, split); pending = pending.slice(split + 1);
      if (!line) throw new Error('Empty speech frame');
      await frame(JSON.parse(line));
     }
     // This bounds a single model frame, not total text or narration length.
     if (pending.length > 16 * 1024 * 1024) throw new Error('Speech frame exceeds transport bound');
     if (result.done) break;
    }
    if (pending || !complete) throw new Error('Speech stream ended without completion');
    check();
    const bytes = samples * 2;
    if (bytes > 0xffffffff - 36) throw new Error('Speech exceeds WAV container size');
    const header = new Uint8Array(44), v = new DataView(header.buffer);
    header.set(new TextEncoder().encode('RIFF'), 0); v.setUint32(4, 36 + bytes, true);
    header.set(new TextEncoder().encode('WAVEfmt '), 8); v.setUint32(16, 16, true);
    v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, 24000, true);
    v.setUint32(28, 48000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    header.set(new TextEncoder().encode('data'), 36); v.setUint32(40, bytes, true);
    transportFinished = true;
    status.textContent = 'Speech generated · first audio ' + (complete.first_audio_ms / 1000).toFixed(2) + ' s';
    status.dispatchEvent(new CustomEvent('myth-stream-complete', { bubbles: true, detail: complete }));
    if (!turn.sources.size) { active = undefined; stop.hidden = true; void turn.context.close(); }
    return new Blob([header, ...pcm], { type: 'audio/wav' });
   } catch (error) {
    const ownsStatus = active === turn || !active;
    if (active === turn) cancel();
    if (ownsStatus) status.textContent = (error as Error).name === 'AbortError' ? 'Speech interrupted' : 'Speech failed';
    throw error;
   } finally { if (reader) { await reader.cancel().catch(() => {}); reader.releaseLock(); } }
  }
 };
}
