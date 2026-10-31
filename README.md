# Myth — native Rust speech on NVIDIA Jetson

Myth is a native Rust speech project for NVIDIA Jetson Orin Nano. This
standalone repository contains the complete native release and four selected
voice recordings, published on 31 October 2026.

- [`rust/`](rust/): complete independent Rust crate, locked dependencies, native
  CUDA/TensorRT C ABI bridge, exporters, regression tests and licence notices.
- [`rust/README.md`](rust/README.md): development history, numerical acceptance
  gates and native service contracts. Earlier dated sections describe earlier
  conversion stages; the final production-launcher section records the cutover.
- [`rust/ui/`](rust/ui/): exact deployed workshop, including bundled local assets.
  [`rust/workshop/`](rust/workshop/) contains its TypeScript additions and build entry.
- [`samples/`](samples/): ships computer, tpol, the doctor and seven originals,
  with source-verified SHA256 values in `samples/manifest.json`.
- [`rust/deployment/`](rust/deployment/): production launcher and catalogue
  examples. Their pinned absolute asset paths describe the original deployment.
- [`rust/public-backup.json`](rust/public-backup.json): source revision and exact
  original native-component checksums. `rust/SHA256SUMS` covers the public export.

Build the standalone executable with `cargo build --manifest-path rust/Cargo.toml
--release --locked`. Build the GPU bridge on the target with `bash
rust/native/build.sh /absolute/path/libchatterbox_trt.so`. Native inference uses
CUDA/TensorRT, cuBLAS, libsndfile and SoXR, plus admitted model/tensor manifests.
Model weights and machine-specific TensorRT engines are separate dependencies;
they are not included in this code-and-samples backup. Build-time exporters and
historical Python parity tools do not make Python a native inference dependency.

The workshop assets are ready to serve. To rebuild its TypeScript additions, run
`npm install --prefix rust/workshop` followed by `node rust/workshop/build.mjs
rust/ui/vendor`. The deployment examples retain the original environment; adapt
and repin paths when restoring elsewhere. This export does not change a running
service. Source licence notices are retained; the reference recordings are not
relicensed under the software licences.

## Project identity

The project is Myth and its repository is `myth-jetson-nano`. This is a fresh
repository containing the native release, without the earlier Python server
or its Git history. Historical identifiers in the native source, compatibility
contracts and deployment snapshots remain unchanged so the release retains
its verified behaviour. Third-party attribution and licence terms are retained.
