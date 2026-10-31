//! Portable S3Gen admission: keep FP32 reference state exact and immutable.
//! Synthetic cases exercise corruption and the odd mel tail retained by embed_ref.
use chatterbox_rs::s3gen::{S3GenFlowAssets, S3GenReference};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

struct Bundle {
    root: PathBuf,
    manifest: Value,
}
impl Drop for Bundle {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}
impl Bundle {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "s3gen-reference-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let mut bundle = Self {
            root,
            manifest: json!({
                "schema_version": 1, "reference_kind": "s3gen_prepared",
                "source_sha256": "a".repeat(64), "prompt_token_len": 2, "prompt_feat_len": null
            }),
        };
        bundle.asset("prompt_tokens", "tokens.json", b"[0,6560]");
        // Deliberately unrepresentable in FP16; values and signed zero must survive.
        bundle.float("prompt_feat", &[1, 5, 80], &vec![1.0000001; 400]);
        let mut speaker = vec![0.25; 192];
        speaker[0] = -0.0;
        speaker[1] = 100000.125;
        bundle.float("embedding", &[1, 192], &speaker);
        bundle
    }
    fn asset(&mut self, key: &str, name: &str, bytes: &[u8]) {
        fs::write(self.root.join(name), bytes).unwrap();
        self.manifest[key] = json!({"file":name, "sha256":sha(bytes)});
    }
    fn float(&mut self, key: &str, shape: &[usize], values: &[f32]) {
        let dims = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dims},), }}\n");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        self.asset(key, &format!("{key}.npy"), &bytes);
    }
    fn load(&self) -> Result<S3GenReference, String> {
        let bytes = serde_json::to_vec(&self.manifest).unwrap();
        let path = self.root.join("s3gen.json");
        fs::write(&path, &bytes).unwrap();
        S3GenReference::open(&path, &sha(&bytes))
    }

    fn flow(&mut self) -> Result<S3GenFlowAssets, String> {
        // Sparse rows make token order and reference/generated boundaries visible.
        let mut table = vec![0.0; 6561 * 512];
        for id in [0, 1, 2, 6560] {
            table[id * 512] = id as f32 + 0.5;
        }
        self.float("token_embedding", &[6561, 512], &table);
        self.float("speaker_weight", &[80, 192], &vec![0.0; 80 * 192]);
        self.float("speaker_bias", &[80], &vec![0.125; 80]);
        let manifest = json!({
            "schema_version": 1, "flow_kind": "s3gen_encoder_input",
            "source_sha256": "b".repeat(64),
            "token_embedding": self.manifest["token_embedding"],
            "speaker_weight": self.manifest["speaker_weight"],
            "speaker_bias": self.manifest["speaker_bias"]
        });
        let bytes = serde_json::to_vec(&manifest).unwrap();
        fs::write(self.root.join("flow.json"), &bytes).unwrap();
        // Flow assets are a separate contract; do not contaminate ref_dict.
        for key in ["token_embedding", "speaker_weight", "speaker_bias"] {
            self.manifest.as_object_mut().unwrap().remove(key);
        }
        S3GenFlowAssets::open(&self.root.join("flow.json"), &sha(&bytes))
    }
}

#[test]
fn s3gen_flow_order_admission_and_immutable_replay() {
    let mut bundle = Bundle::new();
    let assets = bundle.flow().unwrap();
    let reference = bundle.load().unwrap();
    let prepared = assets.prepare(&reference).unwrap();
    let input = prepared.assemble(&[2, 1]).unwrap();
    assert_eq!(input.token_ids(), &[0, 6560, 2, 1]);
    assert_eq!(input.token_len(), 4);
    assert_eq!(input.embeddings().shape, [1, 4, 512]);
    assert_eq!(input.mask().values, [1.0; 4]);
    assert_eq!(input.mask().shape, [1, 4, 1]);
    for (row, expected) in input
        .embeddings()
        .values
        .chunks_exact(512)
        .zip([0.5, 6560.5, 2.5, 1.5])
    {
        assert_eq!(row[0], expected);
        assert!(row[1..].iter().all(|&v| v == 0.0));
    }
    assert_eq!(prepared.speaker().values, vec![0.125; 80]);
    assert_eq!(prepared.reference().prompt_feat().shape, [1, 5, 80]);
    for ids in [vec![], vec![-1], vec![6561], vec![6562], vec![i64::MAX]] {
        assert!(prepared.assemble(&ids).is_err(), "{ids:?}");
    }
    let receipt = serde_json::to_value(input.receipt()).unwrap();
    fs::write(bundle.root.join("token_embedding.npy"), b"changed").unwrap();
    fs::write(bundle.root.join("prompt_feat.npy"), b"changed").unwrap();
    assert_eq!(
        serde_json::to_value(prepared.assemble(&[2, 1]).unwrap().receipt()).unwrap(),
        receipt
    );
    assert_eq!(prepared.assemble(&[1]).unwrap().token_len(), 3);
}

#[test]
fn s3gen_flow_normalization_zero_and_tiny_speaker() {
    let mut bundle = Bundle::new();
    let assets = bundle.flow().unwrap();
    for value in [0.0, 1e-15, 0.5] {
        bundle.float("embedding", &[1, 192], &vec![value; 192]);
        let reference = bundle.load().unwrap();
        let prepared = assets.prepare(&reference).unwrap();
        let expected = if value == 0.0 {
            0.0
        } else if value < 1e-12 {
            value / 1e-12
        } else {
            1.0 / 192.0f32.sqrt()
        };
        assert!(prepared
            .normalized_speaker()
            .values
            .iter()
            .all(|v| (v - expected).abs() < 1e-7));
    }
    bundle.float("embedding", &[1, 192], &vec![1e30; 192]);
    let reference = bundle.load().unwrap();
    assert!(assets.prepare(&reference).is_err());
}

#[test]
fn s3gen_flow_rejects_corrupt_or_incompatible_weights() {
    let mut bundle = Bundle::new();
    bundle.flow().unwrap();
    let path = bundle.root.join("flow.json");
    let original: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    for (key, value) in [
        ("schema_version", json!(2)),
        ("flow_kind", json!("other")),
        ("source_sha256", json!("bad")),
        ("extra", json!(1)),
    ] {
        let mut manifest = original.clone();
        manifest[key] = value;
        let bytes = serde_json::to_vec(&manifest).unwrap();
        fs::write(&path, &bytes).unwrap();
        assert!(S3GenFlowAssets::open(&path, &sha(&bytes)).is_err());
    }
    for (key, shape, values) in [
        ("speaker_weight", vec![192, 80], vec![0.0; 192 * 80]),
        ("speaker_bias", vec![80], vec![f32::INFINITY; 80]),
        ("token_embedding", vec![6560, 512], vec![0.0; 6560 * 512]),
    ] {
        // Isolate each malformed table so a prior checksum failure cannot
        // accidentally satisfy this case before its intended validator runs.
        let mut bundle = Bundle::new();
        bundle.flow().unwrap();
        let path = bundle.root.join("flow.json");
        let original: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        bundle.float(key, &shape, &values);
        let mut manifest = original.clone();
        manifest[key] = bundle.manifest[key].clone();
        let bytes = serde_json::to_vec(&manifest).unwrap();
        fs::write(&path, &bytes).unwrap();
        assert!(S3GenFlowAssets::open(&path, &sha(&bytes)).is_err());
    }
    assert!(S3GenFlowAssets::open(&path, &"0".repeat(64)).is_err());
}

#[test]
#[ignore = "set CHATTERBOX_S3GEN_FLOW_ASSETS to the deployed flow oracle directory"]
fn s3gen_flow_deployed_encoder_boundary_parity() {
    let root = PathBuf::from(std::env::var("CHATTERBOX_S3GEN_FLOW_ASSETS").unwrap());
    let oracle: Value =
        serde_json::from_slice(&fs::read(root.join("oracle.json")).unwrap()).unwrap();
    assert_eq!(oracle["reference_preparations"], 0);
    assert_eq!(oracle["flow"]["conformer_executions"], 0);
    assert_eq!(oracle["cuda_initialized"], false);
    let flow = &oracle["flow"];
    let assets = S3GenFlowAssets::open(
        &root.join(flow["manifest"].as_str().unwrap()),
        flow["manifest_sha256"].as_str().unwrap(),
    )
    .unwrap();
    for case in flow["cases"].as_array().unwrap() {
        let path = root.join(case["manifest"].as_str().unwrap());
        let checksum = case["manifest_sha256"].as_str().unwrap();
        let reference = S3GenReference::open(&path, checksum).unwrap();
        let prepared = assets.prepare(&reference).unwrap();
        let tokens_path = root.join(case["generated"].as_str().unwrap());
        let tokens: Vec<i64> = serde_json::from_slice(&fs::read(&tokens_path).unwrap()).unwrap();
        let input = prepared.assemble(&tokens).unwrap();
        let receipt = serde_json::to_value(input.receipt()).unwrap();
        for key in ["token", "token_len", "mask", "encoder_input"] {
            assert_eq!(receipt[key], case["tensors"][key], "{} {key}", case["name"]);
        }
        for (key, values) in [
            ("speaker", &prepared.speaker().values),
            ("normalized_speaker", &prepared.normalized_speaker().values),
        ] {
            let expected = case["speaker_values"][key].as_array().unwrap();
            assert_eq!(values.len(), expected.len());
            for (&actual, expected) in values.iter().zip(expected) {
                let expected = expected.as_f64().unwrap();
                assert!(
                    (f64::from(actual) - expected).abs() <= 2e-6 + 1e-5 * expected.abs(),
                    "{} {key}: {actual} != {expected}",
                    case["name"]
                );
            }
        }
        assert_eq!(
            serde_json::to_value(prepared.assemble(&tokens).unwrap().receipt()).unwrap(),
            receipt
        );
        let cli = Command::new(env!("CARGO_BIN_EXE_chatterbox-rs"))
            .args([
                "s3-flow-input",
                path.to_str().unwrap(),
                checksum,
                root.join(flow["manifest"].as_str().unwrap())
                    .to_str()
                    .unwrap(),
                flow["manifest_sha256"].as_str().unwrap(),
                tokens_path.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            cli.status.success(),
            "{}",
            String::from_utf8_lossy(&cli.stderr)
        );
        let cli: Value = serde_json::from_slice(&cli.stdout).unwrap();
        assert_eq!(cli["flow_input"], receipt);
        assert_eq!(cli["reference_preparations"], 0);
    }
}

#[test]
fn s3gen_reference_fp32_and_reuse_after_source_change() {
    let bundle = Bundle::new();
    let loaded = bundle.load().unwrap();
    assert_eq!(loaded.prompt_tokens(), &[0, 6560]);
    assert_eq!(loaded.prompt_token_len(), 2);
    assert_eq!(loaded.prompt_feat_len(), None);
    assert_eq!(loaded.prompt_feat().shape, [1, 5, 80]);
    assert!(loaded
        .prompt_feat()
        .values
        .iter()
        .all(|v| v.to_bits() == 1.0000001f32.to_bits()));
    assert_eq!(loaded.embedding().values[0].to_bits(), (-0.0f32).to_bits());
    assert_eq!(loaded.embedding().values[1], 100000.125);
    let before = serde_json::to_value(loaded.receipt()).unwrap();
    fs::write(bundle.root.join("prompt_feat.npy"), b"changed").unwrap();
    assert!(bundle.load().is_err());
    assert_eq!(serde_json::to_value(loaded.receipt()).unwrap(), before);
    assert_eq!(
        loaded.prompt_feat().values[0].to_bits(),
        1.0000001f32.to_bits()
    );
}

#[test]
fn s3gen_reference_rejects_incompatible_state() {
    for (key, value) in [
        ("schema_version", json!(2)),
        ("reference_kind", json!("t3")),
        ("source_sha256", json!("unknown")),
        ("prompt_token_len", json!(1)),
        ("prompt_feat_len", json!(4)),
        ("extra", json!(true)),
    ] {
        let mut bundle = Bundle::new();
        bundle.manifest[key] = value;
        assert!(bundle.load().is_err(), "{key}");
    }
    for tokens in ["[]", "[-1,0]", "[0,6561]", "[0,1.0]", "[0,true]"] {
        let mut bundle = Bundle::new();
        bundle.asset("prompt_tokens", "tokens.json", tokens.as_bytes());
        assert!(bundle.load().is_err(), "{tokens}");
    }
    for (shape, values) in [
        (vec![1, 3, 80], vec![0.0; 240]),
        (vec![2, 5, 80], vec![0.0; 800]),
        (vec![1, 5, 79], vec![0.0; 395]),
        (vec![1, 5, 80], vec![0.0; 399]),
        (vec![1, 5, 80], vec![f32::NAN; 400]),
        (vec![1, 5, 80], vec![f32::INFINITY; 400]),
    ] {
        let mut bundle = Bundle::new();
        bundle.float("prompt_feat", &shape, &values);
        assert!(bundle.load().is_err(), "{shape:?}");
    }
    let mut bundle = Bundle::new();
    bundle.float("embedding", &[1, 256], &vec![0.0; 256]);
    assert!(bundle.load().is_err());
    let mut bundle = Bundle::new();
    bundle.manifest["prompt_feat"]["file"] = json!("../prompt_feat.npy");
    assert!(matches!(bundle.load(),Err(e) if e.contains("filename")));
}

#[test]
fn s3gen_reference_even_mels_and_explicit_feature_length() {
    let mut bundle = Bundle::new();
    bundle.float("prompt_feat", &[1, 4, 80], &vec![0.0; 320]);
    bundle.manifest["prompt_feat_len"] = json!(4);
    let loaded = bundle.load().unwrap();
    assert_eq!(loaded.prompt_feat_len(), Some(4));
    assert_eq!(loaded.receipt().prompt_feat_len.unwrap().shape, [1]);
    let path = bundle.root.join("s3gen.json");
    assert!(S3GenReference::open(&path, &"0".repeat(64)).is_err());
}

#[test]
#[ignore = "set CHATTERBOX_S3GEN_ASSETS to the retained-reference oracle directory"]
fn s3gen_reference_deployed_handoff_parity() {
    let root = PathBuf::from(std::env::var("CHATTERBOX_S3GEN_ASSETS").unwrap());
    let oracle: Value =
        serde_json::from_slice(&fs::read(root.join("oracle.json")).unwrap()).unwrap();
    assert_eq!(oracle["reference_preparations"], 0);
    assert_eq!(oracle["cuda_initialized"], false);
    for case in oracle["cases"].as_array().unwrap() {
        let path = root.join(case["manifest"].as_str().unwrap());
        let expected = case["manifest_sha256"].as_str().unwrap();
        let reference = S3GenReference::open(&path, expected).unwrap();
        // Exact retained bits, plus independent deployed forward interception.
        // Python casts integer IDs/lengths to float before flow casts IDs back;
        // native state keeps IDs integral and checks exact handoff values.
        let receipt = serde_json::to_value(reference.receipt()).unwrap();
        assert_eq!(receipt, case["reference"], "{}", case["name"]);
        assert_eq!(case["handoff_exact_values"], true);
        assert_eq!(receipt, serde_json::to_value(reference.receipt()).unwrap());
        let result = Command::new(env!("CARGO_BIN_EXE_chatterbox-rs"))
            .args(["s3-reference", path.to_str().unwrap(), expected])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let cli: Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(cli["reference"], receipt);
        assert_eq!(cli["reference_preparations"], 0);
        assert!(
            cli["finished_unix_ms"].as_u64().unwrap() >= cli["started_unix_ms"].as_u64().unwrap()
        );
    }
    assert!(Path::new(&root).is_dir());
}
