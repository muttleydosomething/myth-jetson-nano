import { Rnnoise } from '@shiguredo/rnnoise-wasm';

// The pinned RNNoise model accepts 48 kHz mono in signed-16-bit PCM scale.
// Its delayed spectrum plus overlap-add introduces TWO frames of latency.
self.onmessage = async (event: MessageEvent<{ samples: Float32Array; mix: number; gainDb: number }>) => {
 let state: ReturnType<Rnnoise['createDenoiseState']> | undefined;
 try {
  const { samples, mix, gainDb } = event.data;
  if (!samples.length || !Number.isFinite(mix) || mix < 0 || mix > 1) throw new Error('Invalid cleanup input');
  if (!Number.isFinite(gainDb) || gainDb < 0 || gainDb > 24) throw new Error('Invalid sample amplification');
  let output = samples;
  // Amplification alone does not initialise the neural model or alter timing.
  if (mix > 0) {
  const model = await Rnnoise.load();
  state = model.createDenoiseState();
  const frameSize = model.frameSize;
  const delay = 2 * frameSize;
  const frame = new Float32Array(frameSize);
  output = new Float32Array(samples.length);
  // Flush the delayed final speech as well as the partial final input frame.
  for (let offset = 0; offset < samples.length + delay; offset += frameSize) {
   frame.fill(0);
   for (let i = 0; i < frameSize && offset + i < samples.length; i++) {
    const value = samples[offset + i];
    if (!Number.isFinite(value)) throw new Error('Recording contains invalid audio');
    frame[i] = value * 32768;
   }
   state.processFrame(frame);
   for (let i = 0; i < frameSize; i++) {
    const index = offset + i - delay;
    if (index >= 0 && index < output.length) {
     const wet = frame[i] / 32768;
     if (!Number.isFinite(wet)) throw new Error('Cleanup produced invalid audio');
     output[index] = wet * mix + samples[index] * (1 - mix);
    }
   }
  }
  }
  // Measure the unclipped signal. One global gain preserves the waveform; hard
  // clipping or a per-frame limiter would flatten peaks or cause audible pumping.
  let peak = 0;
  for (const value of output) {
   if (!Number.isFinite(value)) throw new Error('Recording contains invalid audio');
   peak = Math.max(peak, Math.abs(value));
  }
  const requestedGain = 10 ** (gainDb / 20);
  const appliedGain = peak > 0 ? Math.min(requestedGain, 1 / peak) : requestedGain;
  for (let i = 0; i < output.length; i++) output[i] *= appliedGain;
  self.postMessage({ samples: output, appliedGainDb: 20 * Math.log10(appliedGain),
   limited: appliedGain + 1e-6 < requestedGain }, { transfer: [output.buffer] });
 } catch (error) {
  self.postMessage({ error: error instanceof Error ? error.message : 'Voice cleanup failed' });
 } finally {
  state?.destroy();
 }
};
