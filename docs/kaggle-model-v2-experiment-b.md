# Model V2 Experiment B: augmentation-only Kaggle workflow

## Status and scientific purpose

Experiment A is the finalized baseline. Experiment B is a separate, reproducible experiment intended to test one hypothesis: whether moderately stronger **TRAIN-only** augmentation improves robustness to real-world image conditions without materially degrading overall VALIDATION performance.

Experiment B is implemented but is **not authorized for training by default**. The dedicated notebook is [`notebooks/kaggle_model_v2_experiment_b.ipynb`](../notebooks/kaggle_model_v2_experiment_b.ipynb). It must remain stopped after the preflight until that report receives separate human review.

INTERNAL TEST and PlantDoc TEST remain locked. Neither may be attached, loaded, predicted, evaluated, or used for checkpoint selection. PlantDoc **TRAIN** remains one of the approved development sources and is not the locked PlantDoc TEST dataset.

## Controlled experiment

The only primary experimental variable is the complete augmentation policy applied dynamically to TRAIN images. No new augmentation family is introduced.

| Setting | Experiment A | Experiment B |
|---|---:|---:|
| Horizontal flip | enabled | enabled |
| Vertical flip | enabled | enabled |
| Rotation | ±15° | **±20°** |
| Horizontal/vertical translation | ±10% | **±12%** |
| Zoom | 0.90–1.10 | **0.85–1.15** |
| Brightness | ±0.10 | **±0.15** |
| Contrast | approximately 0.90–1.10 | **0.85–1.15** |
| Fill mode | reflect | reflect |
| Output clipping | `[0,1]` | `[0,1]` |

The policy explicitly forbids hue/saturation changes, blur, sensor noise, shear, perspective transformations, random crops, MixUp, CutMix, synthetic images, and background replacement. Class weights remain `None`.

The following remain controlled and identical to Experiment A:

- TRAIN manifest: 58,857 images and all 39 classes;
- VALIDATION manifest: 7,362 images and all 39 classes;
- seed `20260810`;
- MobileNetV2 with fresh ImageNet initialization;
- RGB `224×224`, float32, `pixel / 255.0` preprocessing;
- 39-class taxonomy and ordering;
- batch size 32;
- sparse categorical crossentropy and Adam configuration;
- Phase 1 and Phase 2 epoch limits and learning rates;
- Phase 2 boundary at `block_13_expand`;
- all 52 BatchNormalization layers frozen in Phase 2;
- callbacks and candidate tie-break rules;
- overall VALIDATION Macro-F1 for checkpoint selection;
- Python 3.11, TensorFlow/Keras 2.15 and NumPy 1.26.4.

The dedicated policy is `training/config/model-v2-experiment-b-policy.json`. It identifies Experiment A as its baseline and locks the baseline policy plus TRAIN and VALIDATION manifests using platform-independent LF-canonicalized SHA-256 values. The finalized Experiment A policy itself is not modified.

## Isolated Kaggle runtime

The Kaggle system kernel remains an orchestrator. The existing bootstrap creates the pinned ML runtime at:

```text
/kaggle/working/agridiagnose-tf215-runtime/
└── venvs/agridiagnose-tf215/bin/python
```

Experiment B uses `scripts/run_kaggle_model_v2_experiment_b.py` from that interpreter. `MPLBACKEND=Agg` is set for the subprocess so report plots are headless from the beginning.

The notebook pins the reviewed immutable Experiment B implementation commit
`438a170aede710fd3d9ca2cd41815e49849c1744`. Never replace it with a moving
branch name; a future code change requires a new reviewed implementation commit
and an explicit notebook repin.

## Required Kaggle inputs

Attach only the same five private TRAIN/VALIDATION sources used by Experiment A:

| Configuration key | Unique data-root directory |
|---|---|
| `historical` | `historical-mendeley-39` |
| `pldd_up` | `pldd_up` |
| `seasonal_corn` | `seasonal_corn` |
| `plantdoc_train` | `plantdoc-train` |
| `banu_deb` | `potato-banu-deb-originals` |

The shared resolver searches under `/kaggle/input`, supports both direct and nested Kaggle mount layouts, requires exactly one match per source, and rejects escaping or TEST-like paths. Do not hardcode a Kaggle username.

## Safe notebook sequence

1. Enable a Kaggle GPU and Internet.
2. Attach only the five approved development datasets.
3. Pin and clone the reviewed Experiment B implementation commit.
4. Bootstrap the isolated Python 3.11 environment.
5. Run the TensorFlow 2.15 CUDA/GPU smoke gate.
6. Discover and print all five approved source roots.
7. Keep `BATCH_SIZE = 32`, `START_TRAINING = False`, and `RESTART_INTERRUPTED_PHASE = False`.
8. Write `experiment-b-config.json`.
9. Run the exhaustive Experiment B preflight.
10. Download and review the runtime and preflight reports.
11. Stop. Do not authorize training in the same review step.

The preflight command is:

```bash
/kaggle/working/agridiagnose-tf215-runtime/venvs/agridiagnose-tf215/bin/python \
  scripts/run_kaggle_model_v2_experiment_b.py preflight \
  --config /kaggle/working/agridiagnose-tf215-runtime/experiment-b-config.json \
  --output /kaggle/working/agridiagnose-tf215-runtime/experiment-b-preflight.json
```

The expected status is:

```text
KAGGLE_TF215_GPU_EXPERIMENT_B_PREFLIGHT_PASSED
```

The report must show all of the following before any training decision:

- TRAIN `58,857/58,857`, missing `0`, unreadable `0`, coverage `39/39`;
- VALIDATION `7,362/7,362`, missing `0`, unreadable `0`, coverage `39/39`;
- exact TRAIN and VALIDATION hashes matching the Experiment A locks;
- the exact approved Experiment B augmentation layers, order, values, and output clipping;
- VALIDATION augmentation disabled;
- input `(None,224,224,3)` and output `(None,39)`;
- fresh ImageNet initialization and Phase 1 backbone frozen;
- Phase 2 beginning at `block_13_expand`;
- all 52 BatchNormalization layers frozen;
- class weights `null`;
- `training_performed=false`;
- `internal_test_loaded=false` and `plantdoc_test_loaded=false`.

## Dual training authorization

Training is guarded twice:

1. the validated execution config must contain `start_training=true`;
2. the CLI invocation must include `--authorize-training`.

All other combinations stop with `TRAINING_DISABLED_BY_USER`. The notebook exposes two separate booleans, both `False` by default. Preflight review does not imply training authorization.

If a later human decision authorizes Experiment B, the isolated command is conceptually:

```bash
/kaggle/working/agridiagnose-tf215-runtime/venvs/agridiagnose-tf215/bin/python \
  scripts/run_kaggle_model_v2_experiment_b.py train \
  --config /kaggle/working/agridiagnose-tf215-runtime/experiment-b-config.json \
  --authorize-training
```

Do not execute this command during implementation or preflight review.

## Separate output boundaries

Experiment B cannot target an Experiment A output path. Its approved Kaggle paths are:

```text
/kaggle/working/models/candidates/agri-diagnose-v2-exp-b/
/kaggle/working/agridiagnose-exp-b-results/
/kaggle/working/agridiagnose-exp-b-results.zip
```

After separately authorized training, the result package is expected to contain:

- `phase1-history.csv` and `phase2-history.csv`;
- `phase1-best.keras` and `phase2-best.keras` in the candidate directory;
- `agri-diagnose-v2-exp-b.keras`;
- `validation-metrics.json`;
- `validation-confusion-matrix.csv` and `.png`;
- loss, accuracy and Macro-F1 learning-curve PNGs;
- `environment.json`, `experiment.json`, and `preflight.json`;
- `model-v2-exp-b-summary.json` and `model-v2-exp-b-report.md`;
- the final ZIP archive.

Checkpoint selection remains based solely on overall VALIDATION Macro-F1 with the same Experiment A tie-break order. The real-world VALIDATION slice is reported after selection and cannot select a checkpoint.

## A-vs-B comparison

`training.validation_comparison` performs a read-only, **VALIDATION-only** comparison after both experiments have finalized VALIDATION artifacts. It requires supplied Experiment A results; no rounded baseline metric is hardcoded.

The comparison:

- verifies the immutable 7,362-row VALIDATION manifest and prediction order;
- validates both experiment identities and safety metadata;
- rejects INTERNAL TEST or PlantDoc TEST access declarations;
- loads no model or image and performs no inference;
- reports overall accuracy, loss and Macro-F1 deltas;
- reports real-world accuracy and Macro-F1 deltas;
- reports the generalization-gap change;
- reports per-class and aggregate Tomato changes;
- reports Potato Early/Late bidirectional confusion changes;
- ranks major confusion-pair changes;
- performs a fixed-seed, paired, true-class-stratified bootstrap.

This is VALIDATION uncertainty analysis, not TEST evaluation. The Experiment A directory is read-only and hash-checked around report generation. Comparison output must use a separate directory such as:

```text
/kaggle/working/agridiagnose-exp-a-vs-b-validation/
```

The notebook keeps this optional comparison step disabled until finalized A and B directories are explicitly supplied.

Once both VALIDATION result directories exist, the isolated comparison command is:

```bash
/kaggle/working/agridiagnose-tf215-runtime/venvs/agridiagnose-tf215/bin/python \
  scripts/run_kaggle_model_v2_experiment_b.py compare-validation \
  --experiment-a-dir /kaggle/input/REPLACE_WITH_FINALIZED_EXPERIMENT_A_RESULTS \
  --experiment-b-dir /kaggle/working/agridiagnose-exp-b-results \
  --validation-manifest training/datasets/manifests/dataset-v2-validation.csv \
  --output-dir /kaggle/working/agridiagnose-exp-a-vs-b-validation
```

The command reads persisted prediction indices and metadata only. It does not load
either candidate model, any image, INTERNAL TEST, or PlantDoc TEST.

## Current stop condition

Keep all notebook authorization switches `False`. Complete only the runtime gate and Experiment B preflight, preserve their JSON reports, and request a separate human training decision. No INTERNAL TEST or PlantDoc TEST dataset is needed or permitted for this workflow.
