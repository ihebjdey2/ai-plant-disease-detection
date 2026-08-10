# Model V2 Experiment A — GPU-gated preflight

## Status

Experiment A is **not trained** in this environment. TensorFlow 2.15 detects only a CPU and reports no usable GPU. The approved execution gate therefore stopped the run before any call to `model.fit()` with status:

`GPU_NOT_AVAILABLE_FOR_TENSORFLOW`

This document records the completed preflight, not validation performance. No Phase 1/Phase 2 history, validation prediction, confusion matrix, candidate model, or Experiment A performance claim exists yet.

## Environment

- OS: Windows 10, build 10.0.26200, AMD64;
- Python: CPython 3.11.9;
- TensorFlow: 2.15.0, CPU build (`built_with_cuda=false`);
- Keras: 2.15.0;
- NumPy: 1.26.4;
- scikit-learn: 1.4.1.post1;
- physical TensorFlow devices: `/physical_device:CPU:0` only;
- GPU model/memory: unavailable;
- CPU: 12th Gen Intel Core i7-1255U, 12 logical processors;
- RAM: 16,835,268,608 bytes;
- experiment seed: 20260810;
- TensorFlow operation determinism: enabled for preflight;
- bit-identical GPU training guarantee: false.

The environment metadata contains no username, private path, secret, or environment-variable value.

## Exhaustive data preflight

The preflight used only `dataset-v2-train.csv` and `dataset-v2-validation.csv`. It did not open the internal TEST manifest as a dataset; only its file bytes were hashed.

| Partition | Expected | Resolved | Missing | Unreadable | Coverage |
|---|---:|---:|---:|---:|---:|
| TRAIN | 58,857 | 58,857 | 0 | 0 | 39/39 |
| VALIDATION | 7,362 | 7,362 | 0 | 0 | 39/39 |

PlantDoc and Banu/Deb immutable source paths are resolved through their committed audit-manifest `local_file` aliases. No source membership is changed. The resolver also supports Windows extended-length paths after its existing anti-traversal check; this was necessary for 549 Tomato Spider mites paths and one Seasonal Corn filename that exist on NTFS but exceed the classic 260-character API limit.

Tensor checks on deterministic batches passed:

- TRAIN: `(32,224,224,3)`, float32, `[0,1]`, labels `0..38`, augmentation enabled;
- VALIDATION: `(32,224,224,3)`, float32, `[0,1]`, labels `0..38`, augmentation disabled, shuffle disabled.

The preflight caught a slight contrast-augmentation overshoot (`1.00795054`) before training. The dynamic TRAIN pipeline now clips post-augmentation values to `[0,1]`; the final repeated audit reached exactly `[0,1]`.

## MacroF1

`training.metrics.MacroF1` accumulates one 39×39 confusion matrix across all batches and computes the epoch-level mean of 39 per-class F1 scores. It does not average batch-level F1 values.

The deterministic preflight comparison produced:

- TensorFlow MacroF1: 0.8620201349;
- scikit-learn MacroF1: 0.8620201640;
- absolute difference: 0.000000029062;
- tolerance: 0.000001;
- `reset_state()` result: 0;
- result: passed.

Synthetic tests also cover perfect, fully wrong, imbalanced, rare-class, missing-prediction, mixed multiclass, all-39-class, and reset scenarios.

## Architecture and parameter audits

The fresh preflight model uses downloaded official ImageNet MobileNetV2 weights, not `plant_disease_model.h5`:

`MobileNetV2 → GlobalAveragePooling2D → Dropout(0.25) → Dense(39, softmax)`

Phase 1 planned audit:

- output: `(None,39)`;
- total parameters: 2,307,943;
- trainable parameters: 49,959;
- non-trainable parameters: 2,257,984;
- backbone trainable: false;
- Adam learning rate: 1e-3;
- metrics: accuracy, macro_f1.

Phase 2 planned audit:

- total parameters: 2,307,943;
- trainable parameters: 1,713,319;
- non-trainable parameters: 594,624;
- MobileNetV2 layers: 154;
- boundary: `block_13_expand`, runtime index 116;
- first trainable backbone layer: `block_13_expand`;
- trainable backbone layers: 25;
- frozen backbone layers: 129;
- BatchNormalization layers/frozen: 52/52;
- new Adam optimizer planned at 2e-5.

The checkpoint policy monitors `val_macro_f1` in maximum mode. Experiment A class weights are explicitly `None`.

## Locked artifacts

- TRAIN manifest SHA-256: `935483f8bde5596e56aec5ac59e3d032128e685247a59a7a49e29c0ba32c74a3`;
- VALIDATION manifest SHA-256: `fbe6be537c394c907f473401a856650764f2206b0ef59401460e33fe9a432d85`;
- INTERNAL TEST manifest SHA-256: `f0df59c42268163d485feea0e54dd7780aa56fe08a7984ae7869a09c604a9151` (unchanged);
- V1 production model SHA-256: `bf076af34eba83053a7843dfc8b27d77e77412b7629e48227b8e30f278e81d09` (unchanged).

INTERNAL TEST loaded/evaluated: false/false. PlantDoc TEST loaded/evaluated: false/false. Confidence threshold changed: false. Candidate model created: false.

## Next decision

Full CPU training was not launched silently. Experiment A requires a human-approved TensorFlow GPU environment or an explicit change to the execution gate. Package/CUDA changes must not be performed automatically. Once a usable GPU is available, rerun the exact preflight, then implement/execute Phase 1 and Phase 2 without changing architecture, augmentation, class weights, seed, or holdout policy.
