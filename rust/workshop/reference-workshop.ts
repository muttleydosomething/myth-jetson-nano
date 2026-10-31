import { isolateReference } from './reference-isolation';

// Both existing Import buttons enter one draft editor. Only Apply writes to
// Borg's persistent voices volume, through the existing library upload endpoint.
export function setupReferenceWorkshop({ inputs, onApplied }: {
 inputs: HTMLInputElement[]; onApplied: (result: any, selected: string) => void;
}) {
 const panel = document.getElementById('reference-workshop')!;
 const source = panel.querySelector<HTMLSelectElement>('#reference-source')!;
 const mode = panel.querySelector<HTMLSelectElement>('#reference-isolation')!;
 const gain = panel.querySelector<HTMLInputElement>('#reference-gain')!;
 const gainLabel = panel.querySelector<HTMLOutputElement>('#reference-gain-value')!;
 const choice = panel.querySelector<HTMLSelectElement>('#reference-choice')!;
 const original = panel.querySelector<HTMLAudioElement>('#reference-original')!;
 const processed = panel.querySelector<HTMLAudioElement>('#reference-processed')!;
 const preview = panel.querySelector<HTMLButtonElement>('#reference-preview')!;
 const cancel = panel.querySelector<HTMLButtonElement>('#reference-cancel')!;
 const apply = panel.querySelector<HTMLButtonElement>('#reference-apply')!;
 const status = panel.querySelector<HTMLElement>('#reference-status')!;
 let files: File[] = [];
 let cleaned: Blob | null = null;
 let controller: AbortController | null = null;
 let writing = false;
 let processedName = '';
 const urls = new Map<HTMLAudioElement, string>();
 function audio(element: HTMLAudioElement, blob?: Blob) {
  element.pause();
  if (urls.has(element)) URL.revokeObjectURL(urls.get(element)!);
  urls.delete(element);
  element.removeAttribute('src');
  element.hidden = !blob;
  if (blob) { const url = URL.createObjectURL(blob); urls.set(element, url); element.src = url; }
 }
 function invalidate() {
  controller?.abort(); controller = null; cleaned = null;
  audio(processed); choice.value = 'original'; choice.options[1].disabled = true;
  cancel.hidden = true; preview.disabled = false; apply.disabled = !files.length;
  status.textContent = '';
 }
 function selectSource() {
  invalidate();
  audio(original, files[Number(source.value)]);
 }
 for (const input of inputs) input.addEventListener('change', () => {
  if (writing) return;
  files = [...(input.files || [])]; input.value = '';
  source.replaceChildren(...files.map((file, index) => new Option(file.name, String(index))));
  panel.hidden = !files.length;
  mode.value = 'off'; gain.value = '0'; gainLabel.value = '+0 dB';
  selectSource();
 });
 source.addEventListener('change', selectSource);
 mode.addEventListener('change', invalidate);
 gain.addEventListener('input', () => { gainLabel.value = '+' + gain.value + ' dB'; invalidate(); });
 cancel.addEventListener('click', invalidate);
 preview.addEventListener('click', async () => {
  const file = files[Number(source.value)];
  if (!file || writing) return;
  invalidate();
  const request = new AbortController(); controller = request;
  cancel.hidden = false; preview.disabled = true; apply.disabled = true;
  status.textContent = 'Processing sample…';
  try {
   const result = await isolateReference(file, mode.value as 'off' | 'gentle' | 'full', request.signal, Number(gain.value));
   if (controller !== request) return;
   cleaned = result.audio; audio(processed, cleaned);
   // Unique identity avoids the existing endpoint's duplicate-name shortcut.
   processedName = file.name.replace(/\.(wav|mp3)$/i, '').slice(0, 70) + '-processed-' + crypto.randomUUID() + '.wav';
   choice.options[1].disabled = false; choice.value = 'processed';
   status.textContent = result.limited
    ? 'Peak protection limited amplification to ' + result.appliedGainDb.toFixed(1) + ' dB.'
    : 'Preview ready. Amplification: +' + gain.value + ' dB.';
  } catch (error) {
   if (controller !== request) return;
   status.textContent = 'Preview failed: ' + (error instanceof Error ? error.message : 'Unsupported audio');
  } finally {
   if (controller === request) { controller = null; cancel.hidden = true; preview.disabled = false; apply.disabled = false; }
  }
 });
 apply.addEventListener('click', async () => {
  if (writing || !files.length || controller) return;
  const selected = files[Number(source.value)];
  const useProcessed = choice.value === 'processed';
  if (useProcessed && !cleaned) return;
  writing = true;
  // Freeze the draft while committing: a late result cannot consume another input.
  const controls = [...panel.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>('input,select,button'), ...inputs];
  controls.forEach(control => control.disabled = true);
  try {
   const body = new FormData();
   for (const file of files) body.append('files', file);
   if (useProcessed) body.append('files', new File([cleaned!], processedName, { type: 'audio/wav' }));
   const response = await fetch('/upload_predefined_voice', { method: 'POST', body });
   const result = await response.json();
   if (!response.ok || result.errors?.length) throw new Error(result.errors?.map((e: any) => e.error).join('; ') || result.detail || 'Upload failed');
   const name = useProcessed ? processedName : selected.name;
   if (!result.uploaded_files?.includes(name)) throw new Error('Borg did not confirm the selected recording');
   onApplied(result, name);
   files = []; invalidate(); audio(original); panel.hidden = true;
  } catch (error) {
   // The server may have accepted some originals. Keep the draft for correction;
   // its duplicate-name handling makes the next explicit Apply safe to retry.
   status.textContent = 'Apply failed; recording remains pending: ' + (error instanceof Error ? error.message : 'Upload failed');
  } finally {
   writing = false; controls.forEach(control => control.disabled = false);
   apply.disabled = !files.length;
  }
 });
 window.addEventListener('pagehide', () => { invalidate(); audio(original); });
}
