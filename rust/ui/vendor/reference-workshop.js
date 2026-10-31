// borg/chatterbox/patches/myth-reference/reference-isolation.ts
async function isolateReference(file, mode, signal, gainDb = 0) {
  const cancelled = () => {
    if (signal.aborted) throw new DOMException("Cleanup cancelled", "AbortError");
  };
  cancelled();
  const context = new OfflineAudioContext(1, 1, 48e3);
  const decoded = await context.decodeAudioData(await file.arrayBuffer());
  cancelled();
  if (!decoded.length) throw new Error("Recording is empty");
  const samples = new Float32Array(decoded.length);
  for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
    const pcm = decoded.getChannelData(channel);
    for (let i = 0; i < samples.length; i++) samples[i] += pcm[i] / decoded.numberOfChannels;
  }
  const worker = new Worker(new URL("./reference-isolation.worker.js", import.meta.url), { type: "module" });
  const cleaned = await new Promise((resolve, reject) => {
    const dispose = () => {
      signal.removeEventListener("abort", abort);
      worker.terminate();
    };
    const abort = () => {
      dispose();
      reject(new DOMException("Cleanup cancelled", "AbortError"));
    };
    signal.addEventListener("abort", abort, { once: true });
    worker.onmessage = (event) => {
      dispose();
      if (event.data.error || !event.data.samples) reject(new Error(event.data.error || "Cleanup returned no audio"));
      else resolve({ samples: event.data.samples, appliedGainDb: event.data.appliedGainDb, limited: event.data.limited });
    };
    worker.onerror = () => {
      dispose();
      reject(new Error("Voice cleanup processor failed to load"));
    };
    try {
      worker.postMessage({ samples, mix: mode === "off" ? 0 : mode === "gentle" ? 0.65 : 1, gainDb }, [samples.buffer]);
    } catch (error) {
      dispose();
      reject(error);
    }
  });
  cancelled();
  return { audio: pcmWav(cleaned.samples), appliedGainDb: cleaned.appliedGainDb, limited: cleaned.limited };
}
function pcmWav(samples) {
  const byteLength = samples.length * 2;
  if (byteLength > 4294967295 - 36) throw new Error("Cleaned sample exceeds WAV format capacity; the original can still be uploaded");
  const data = new ArrayBuffer(44 + byteLength);
  const view = new DataView(data);
  const text = (offset, value) => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
  };
  text(0, "RIFF");
  view.setUint32(4, 36 + byteLength, true);
  text(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, 48e3, true);
  view.setUint32(28, 96e3, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, byteLength, true);
  for (let i = 0; i < samples.length; i++) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
  }
  return new Blob([data], { type: "audio/wav" });
}

// borg/chatterbox/patches/myth-reference/reference-workshop.ts
function setupReferenceWorkshop({ inputs, onApplied }) {
  const panel = document.getElementById("reference-workshop");
  const source = panel.querySelector("#reference-source");
  const mode = panel.querySelector("#reference-isolation");
  const gain = panel.querySelector("#reference-gain");
  const gainLabel = panel.querySelector("#reference-gain-value");
  const choice = panel.querySelector("#reference-choice");
  const original = panel.querySelector("#reference-original");
  const processed = panel.querySelector("#reference-processed");
  const preview = panel.querySelector("#reference-preview");
  const cancel = panel.querySelector("#reference-cancel");
  const apply = panel.querySelector("#reference-apply");
  const status = panel.querySelector("#reference-status");
  let files = [];
  let cleaned = null;
  let controller = null;
  let writing = false;
  let processedName = "";
  const urls = /* @__PURE__ */ new Map();
  function audio(element, blob) {
    element.pause();
    if (urls.has(element)) URL.revokeObjectURL(urls.get(element));
    urls.delete(element);
    element.removeAttribute("src");
    element.hidden = !blob;
    if (blob) {
      const url = URL.createObjectURL(blob);
      urls.set(element, url);
      element.src = url;
    }
  }
  function invalidate() {
    controller?.abort();
    controller = null;
    cleaned = null;
    audio(processed);
    choice.value = "original";
    choice.options[1].disabled = true;
    cancel.hidden = true;
    preview.disabled = false;
    apply.disabled = !files.length;
    status.textContent = "";
  }
  function selectSource() {
    invalidate();
    audio(original, files[Number(source.value)]);
  }
  for (const input of inputs) input.addEventListener("change", () => {
    if (writing) return;
    files = [...input.files || []];
    input.value = "";
    source.replaceChildren(...files.map((file, index) => new Option(file.name, String(index))));
    panel.hidden = !files.length;
    mode.value = "off";
    gain.value = "0";
    gainLabel.value = "+0 dB";
    selectSource();
  });
  source.addEventListener("change", selectSource);
  mode.addEventListener("change", invalidate);
  gain.addEventListener("input", () => {
    gainLabel.value = "+" + gain.value + " dB";
    invalidate();
  });
  cancel.addEventListener("click", invalidate);
  preview.addEventListener("click", async () => {
    const file = files[Number(source.value)];
    if (!file || writing) return;
    invalidate();
    const request = new AbortController();
    controller = request;
    cancel.hidden = false;
    preview.disabled = true;
    apply.disabled = true;
    status.textContent = "Processing sample\u2026";
    try {
      const result = await isolateReference(file, mode.value, request.signal, Number(gain.value));
      if (controller !== request) return;
      cleaned = result.audio;
      audio(processed, cleaned);
      processedName = file.name.replace(/\.(wav|mp3)$/i, "").slice(0, 70) + "-processed-" + crypto.randomUUID() + ".wav";
      choice.options[1].disabled = false;
      choice.value = "processed";
      status.textContent = result.limited ? "Peak protection limited amplification to " + result.appliedGainDb.toFixed(1) + " dB." : "Preview ready. Amplification: +" + gain.value + " dB.";
    } catch (error) {
      if (controller !== request) return;
      status.textContent = "Preview failed: " + (error instanceof Error ? error.message : "Unsupported audio");
    } finally {
      if (controller === request) {
        controller = null;
        cancel.hidden = true;
        preview.disabled = false;
        apply.disabled = false;
      }
    }
  });
  apply.addEventListener("click", async () => {
    if (writing || !files.length || controller) return;
    const selected = files[Number(source.value)];
    const useProcessed = choice.value === "processed";
    if (useProcessed && !cleaned) return;
    writing = true;
    const controls = [...panel.querySelectorAll("input,select,button"), ...inputs];
    controls.forEach((control) => control.disabled = true);
    try {
      const body = new FormData();
      for (const file of files) body.append("files", file);
      if (useProcessed) body.append("files", new File([cleaned], processedName, { type: "audio/wav" }));
      const response = await fetch("/upload_predefined_voice", { method: "POST", body });
      const result = await response.json();
      if (!response.ok || result.errors?.length) throw new Error(result.errors?.map((e) => e.error).join("; ") || result.detail || "Upload failed");
      const name = useProcessed ? processedName : selected.name;
      if (!result.uploaded_files?.includes(name)) throw new Error("Borg did not confirm the selected recording");
      onApplied(result, name);
      files = [];
      invalidate();
      audio(original);
      panel.hidden = true;
    } catch (error) {
      status.textContent = "Apply failed; recording remains pending: " + (error instanceof Error ? error.message : "Upload failed");
    } finally {
      writing = false;
      controls.forEach((control) => control.disabled = false);
      apply.disabled = !files.length;
    }
  });
  window.addEventListener("pagehide", () => {
    invalidate();
    audio(original);
  });
}
export {
  setupReferenceWorkshop
};
