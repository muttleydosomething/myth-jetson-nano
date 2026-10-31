// Focused browser proof runs the real worker and Apply UI; local mode fakes only
// the library transaction. Pass a live URL to exercise Borg's existing endpoint.
import { createRequire } from 'node:module';
import { createServer } from 'node:http';
import { readFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const require = createRequire(path.resolve('frontend/hailstorm-frontend/package.json'));
const { chromium, expect } = require('@playwright/test');
const [site, proof] = process.argv.slice(2);
await mkdir(proof, { recursive: true });
const live = site.startsWith('https:');
let server;
let url = site;
if (!live) {
 server = createServer(async (req, res) => {
  try {
   const file = path.join(site, req.url === '/' ? 'index.html' : req.url.split('?')[0]);
   const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
   res.setHeader('Content-Type', types[path.extname(file)] || 'application/octet-stream');
   res.end(await readFile(file));
  } catch { res.writeHead(404); res.end(); }
 });
 await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
 url = 'http://127.0.0.1:' + server.address().port;
}
const browser = await chromium.launch({ headless: true });
try {
 const page = await browser.newPage({ viewport: { width: 1440, height: 1200 } });
 const errors = []; page.on('pageerror', e => errors.push(e.message));
 const uploads = [];
 if (!live) {
  await page.route('**/api/ui/initial-data', route => route.fulfill({ json: { config: { ui_state: {}, generation_defaults: {} }, predefined_voices: [], reference_files: [], presets: [] } }));
  await page.route('**/save_settings', route => route.fulfill({ json: {} }));
  await page.route('**/upload_predefined_voice', route => {
   const body = route.request().postDataBuffer().toString('latin1');
   const names = [...body.matchAll(/filename="([^"]+)"/g)].map(m => m[1]);
   uploads.push(names);
   route.fulfill({ json: { uploaded_files: names, errors: [], all_predefined_voices: names.map(filename => ({ filename, display_name: filename })) } });
  });
 }
 await page.goto(url);
 await expect(page.locator('#predefined-voice-import-button')).toBeVisible();
 const wav = Buffer.alloc(44 + 96037 * 2);
 wav.write('RIFF'); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
 wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
 wav.writeUInt32LE(48000, 24); wav.writeUInt32LE(96000, 28);
 wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34); wav.write('data', 36); wav.writeUInt32LE(wav.length - 44, 40);
 let seed = 42;
 for (let i = 44; i < wav.length; i += 2) { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; wav.writeInt16LE(Math.round((seed / 4294967296 - .5) * 6000), i); }
 const filename = 'myth-reference-proof-' + Date.now() + '.wav';
 await page.locator('#predefined-voice-file-input').setInputFiles({ name: filename, mimeType: 'audio/wav', buffer: wav });
 await expect(page.locator('#reference-workshop')).toBeVisible();
 await page.locator('#reference-gain').fill('6');
 await page.locator('#reference-preview').click();
 await expect(page.locator('#reference-processed')).toBeVisible();
 const error = await page.locator('#reference-processed').evaluate(async audio => {
  const context = new OfflineAudioContext(1, 1, 48000);
  const decode = async element => (await context.decodeAudioData(await (await fetch(element.src)).arrayBuffer())).getChannelData(0);
  const [before, after] = await Promise.all([decode(document.querySelector('#reference-original')), decode(audio)]);
  if (before.length !== after.length) return Infinity;
  let error = 0; for (let i = 0; i < before.length; i++) error = Math.max(error, Math.abs(after[i] - before[i] * 10 ** (6 / 20)));
  return error;
 });
 assert(error < .00004, 'amplification preserves waveform and duration');
 await page.locator('#reference-gain').fill('24');
 await expect(page.locator('#reference-processed')).toBeHidden();
 await page.locator('#reference-preview').click();
 await expect(page.locator('#reference-status')).toContainText('Peak protection');
 await page.locator('#reference-isolation').selectOption('full');
 await page.locator('#reference-gain').fill('6');
 await page.locator('#reference-preview').click();
 await expect(page.locator('#reference-processed')).toBeVisible();
 const energy = await page.locator('#reference-processed').evaluate(async audio => {
  const context = new OfflineAudioContext(1, 1, 48000);
  const pcm = (await context.decodeAudioData(await (await fetch(audio.src)).arrayBuffer())).getChannelData(0);
  let sum = 0; for (const value of pcm) sum += value * value; return sum / pcm.length;
 });
 assert(energy < .0004, 'RNNoise reduces broadband noise');
 assert.equal(uploads.length, 0, 'preview never uploads');
 await page.locator('#reference-workshop').screenshot({ path: path.join(proof, live ? 'myth-live.png' : 'myth-candidate.png') });
 await page.locator('#reference-apply').click();
 await expect(page.locator('#reference-workshop')).toBeHidden();
 const selected = await page.locator('#predefined-voice-select').inputValue();
 assert.match(selected, /-processed-/);
 if (!live) {
  assert.equal(uploads.length, 1); assert.equal(uploads[0].length, 2);
  // A decode failure keeps the original applicable and a changed source cancels
  // an in-flight worker. Neither operation commits a draft.
  await page.locator('#predefined-voice-file-input').setInputFiles({ name: 'invalid.wav', mimeType: 'audio/wav', buffer: Buffer.from('invalid') });
  await page.locator('#reference-preview').click();
  await expect(page.locator('#reference-status')).toContainText('Preview failed');
  await expect(page.locator('#reference-apply')).toBeEnabled();
  await expect(page.locator('#reference-choice')).toHaveValue('original');
  await page.locator('#predefined-voice-file-input').setInputFiles({ name: 'first.wav', mimeType: 'audio/wav', buffer: wav });
  await page.locator('#reference-isolation').selectOption('gentle');
  await page.locator('#reference-preview').click();
  await page.locator('#predefined-voice-file-input').setInputFiles({ name: 'second.wav', mimeType: 'audio/wav', buffer: wav });
  await expect(page.locator('#reference-processed')).toBeHidden();
  await expect(page.locator('#reference-source')).toContainText('second.wav');
  assert.equal(uploads.length, 1);
 }
 assert.deepEqual(errors, []);
 console.log(JSON.stringify({ status: 'passed', live, filename, selected, amplificationMaxError: error, cleanedEnergy: energy, browserErrors: errors }));
} finally { await browser.close(); if (server) await new Promise(resolve => server.close(resolve)); }
