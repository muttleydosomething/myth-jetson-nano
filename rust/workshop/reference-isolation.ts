export type IsolationMode = 'gentle' | 'full';
export interface ReferenceSampleResult { audio: Blob; appliedGainDb: number; limited: boolean }

// Decode only on explicit Preview. Ordinary uploads keep their existing streaming
// route and have no new duration/size restriction. A worker keeps inference off UI.
export async function isolateReference(file: File, mode: IsolationMode | 'off', signal: AbortSignal, gainDb = 0): Promise<ReferenceSampleResult> {
 const cancelled = () => {
  if (signal.aborted) throw new DOMException('Cleanup cancelled', 'AbortError');
 };
 cancelled();
 const context = new OfflineAudioContext(1, 1, 48000);
 const decoded = await context.decodeAudioData(await file.arrayBuffer());
 cancelled();
 if (!decoded.length) throw new Error('Recording is empty');
 // decodeAudioData resamples to 48 kHz; equal channel mixing matches mono reference
 // admission, without an extra live AudioContext or an audio-device unlock.
 const samples = new Float32Array(decoded.length);
 for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
  const pcm = decoded.getChannelData(channel);
  for (let i = 0; i < samples.length; i++) samples[i] += pcm[i] / decoded.numberOfChannels;
 }
 const worker = new Worker(new URL('./reference-isolation.worker.js', import.meta.url), { type: 'module' });
 const cleaned = await new Promise<{ samples: Float32Array; appliedGainDb: number; limited: boolean }>((resolve, reject) => {
  const dispose = () => { signal.removeEventListener('abort', abort); worker.terminate(); };
  const abort = () => { dispose(); reject(new DOMException('Cleanup cancelled', 'AbortError')); };
  signal.addEventListener('abort', abort, { once: true });
  worker.onmessage = (event: MessageEvent<{ samples?: Float32Array; error?: string; appliedGainDb: number; limited: boolean }>) => {
   dispose();
   if (event.data.error || !event.data.samples) reject(new Error(event.data.error || 'Cleanup returned no audio'));
   else resolve({ samples: event.data.samples, appliedGainDb: event.data.appliedGainDb, limited: event.data.limited });
  };
  worker.onerror = () => { dispose(); reject(new Error('Voice cleanup processor failed to load')); };
  try { worker.postMessage({ samples, mix: mode === 'off' ? 0 : mode === 'gentle' ? 0.65 : 1, gainDb }, [samples.buffer]); }
  catch (error) { dispose(); reject(error); }
 });
 cancelled();
 return { audio: pcmWav(cleaned.samples), appliedGainDb: cleaned.appliedGainDb, limited: cleaned.limited };
}

function pcmWav(samples: Float32Array): Blob {
 // Reference WAV uses the service's existing uncompressed PCM admission. Reject
 // RIFF overflow explicitly rather than truncating a length into 32 bits.
 const byteLength = samples.length * 2;
 if (byteLength > 0xffffffff - 36) throw new Error('Cleaned sample exceeds WAV format capacity; the original can still be uploaded');
 const data = new ArrayBuffer(44 + byteLength);
 const view = new DataView(data);
 const text = (offset: number, value: string) => {
  for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
 };
 text(0, 'RIFF'); view.setUint32(4, 36 + byteLength, true); text(8, 'WAVEfmt ');
 view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
 view.setUint32(24, 48000, true); view.setUint32(28, 96000, true);
 view.setUint16(32, 2, true); view.setUint16(34, 16, true);
 text(36, 'data'); view.setUint32(40, byteLength, true);
 for (let i = 0; i < samples.length; i++) {
  const value = Math.max(-1, Math.min(1, samples[i]));
  view.setInt16(44 + i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
 }
 return new Blob([data], { type: 'audio/wav' });
}
