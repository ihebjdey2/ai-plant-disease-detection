# Model V2 Experiment B: augmentation-only Kaggle workflow

## Status and scientific purpose

Experiment A is the finalized baseline. Experiment B is a separate, reproducible experiment intended to test one hypothesis: whether moderately stronger **TRAIN-only** augmentation improves robustness to real-world image conditions without materially degrading overall VALIDATION performance.

Experiment B is implemented but is **not authorized for training by default**. A previous, explicitly authorized Kaggle run completed Phase 1 epochs 1 through 5 before interruption; epoch 5 is the recorded VALIDATION Macro-F1 checkpoint. The checkpoint, history, runtime audit, and preflight remain preserved. This implementation adds a fail-closed continuation path; it does not itself resume or retrain anything.

The dedicated notebook is [`notebooks/kaggle_model_v2_experiment_b.ipynb`](../notebooks/kaggle_model_v2_experiment_b.ipynb). Its committed values remain `START_TRAINING = False`, `AUTHORIZE_TRAINING_CLI = False`, and `INTERRUPTED_PHASE_ACTION = 'fail'` until a separate human review authorizes one execution.

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

The notebook pins the reviewed immutable Experiment B resume implementation commit
`083ecde28ebf052764812cb3f317532a231d542c`. Never replace it with a moving
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
7. Keep `BATCH_SIZE = 32`, `START_TRAINING = False`, and `INTERRUPTED_PHASE_ACTION = 'fail'`.
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

## Safe exact-enough interrupted-phase recovery

Experiment B uses an explicit `interrupted_phase_action` enum:

- `fail` is the default and refuses any interrupted phase artifacts;
- `resume` continues only after every compatibility proof succeeds;
- `restart` deliberately begins that phase at epoch 0 and retains the historical explicit-restart behavior. It must never be selected as an implicit fallback from a failed resume and may replace that phase's artifacts when training is separately authorized.

The legacy Boolean is accepted only for old Experiment B configuration files: exact `false` maps to `fail` and exact `true` maps to `restart`. A string such as `"false"`, an unknown action, or a file containing both forms is rejected. Experiment A configuration behavior is unchanged.

For the current Phase 1 artifacts, a separately authorized recovery must set all three controls deliberately:

```python
START_TRAINING = True
AUTHORIZE_TRAINING_CLI = True
INTERRUPTED_PHASE_ACTION = 'resume'
```

These values are instructions for the reviewed Kaggle session, not notebook defaults. Before any resumed `model.fit()` call, the runner:

1. reads `phase1-history.csv` and requires its exact schema, finite metrics, and contiguous absolute epochs `1..5`;
2. recomputes the first strict maximum of `val_macro_f1`, matching `ModelCheckpoint(save_best_only=True, mode='max')`, and requires it to be epoch 5;
3. additionally requires epoch 5 to hold the best recorded `val_loss`, because the historical run did not persist an earlier `EarlyStopping.restore_best_weights` buffer;
4. validates the preserved preflight, runtime, Python/TensorFlow/Keras/NumPy/GPU identity, immutable Experiment B policy, TRAIN and VALIDATION manifest hashes, 39-class taxonomy, model phase audits, and explicit false TEST flags;
5. loads `phase1-best.keras` with compile state, verifies its SHA-256/size/mtime remain unchanged while reading it, and rejects incompatible model name, input/output shapes, parameter counts, MobileNetV2 trainability, BatchNormalization policy, loss, Adam configuration, optimizer slots, learning rate, or iteration count;
6. restores the serialized model and Adam state without recompiling, reconstructs the EarlyStopping and ReduceLROnPlateau counters, seeds ModelCheckpoint with the historical best Macro-F1, and invokes Keras with `initial_epoch=5` and `epochs=10`.

The next visible progress line is therefore `Epoch 6/10`. Resume never silently falls back to `initial_epoch=0`.

The resume history callback preloads the original five rows. At each newly completed epoch it requires the next absolute epoch number, writes all old and new rows to a temporary CSV, and atomically replaces the history file. The resulting sequence is `1,2,3,4,5,6,...`, with no duplicated or truncated original rows. A completion marker records whether `model.fit()` returned after reaching the maximum or via EarlyStopping, plus locked history/checkpoint identities.

Phase 2 follows the same rules. It may start normally only after Phase 1 is proven complete. If Phase 2 has interrupted history and a compatible `phase2-best.keras`, it restores that full checkpoint and uses its completed epoch count as `initial_epoch`; partial, mismatched, or TEST-tainted artifacts stop the run. Candidate selection remains overall VALIDATION Macro-F1 only.

This is deliberately described as **exact-enough**, not bit-identical. The historical checkpoint preserves model and optimizer state, but the original `tf.data` shuffle, augmentation, and Dropout random-number streams were not checkpointed. The resumed run therefore cannot reproduce the exact uninterrupted sample/transformation sequence. No policy, architecture, threshold, split, TEST lock, or candidate-selection rule is changed.

Resume fails closed for missing or malformed history, missing or unreadable checkpoints, gaps/duplicates/non-finite values, completed epochs at or above a resumable phase maximum, a selected Macro-F1 epoch other than the last completed epoch, an unreconstructable EarlyStopping state, policy/manifest/taxonomy/runtime/model/optimizer mismatches, Phase 2 artifacts beside an interrupted Phase 1, and any INTERNAL TEST or PlantDoc TEST safety violation.

## Optional private Kaggle Dataset persistence

`/kaggle/working` is ephemeral. Experiment B can therefore version the minimum
recovery state to an existing **private** Kaggle Dataset after every completed
epoch. Persistence is execution infrastructure only; it does not change the
training policy, callbacks' scientific behavior, checkpoint selection, model,
data, metrics, or TEST locks.

The committed execution defaults are:

```python
PERSISTENT_BACKUP_ENABLED = False
PERSISTENT_BACKUP_DATASET_HANDLE = ''
```

After separately creating a private Dataset and reviewing its permissions, an
operator may configure an `owner/dataset` handle. The handle is never stored in
the scientific policy. `kagglehub.dataset_upload` versions a dedicated staging
directory after the local history and ModelCheckpoint callbacks have completed.
The version note identifies Experiment B, the phase, and the absolute completed
epoch.

Only these files can be staged when available:

- `phase1-best.keras`, `phase1-history.csv`, `phase1-complete.json`;
- `phase2-best.keras`, `phase2-history.csv`, `phase2-complete.json`;
- `environment-runtime.json` and `preflight.json`.

Phase 2 versions retain both phase lineages. Images, manifests, temporary files,
project sources, and unrelated outputs never enter the staging directory. If
staging, byte verification, authentication, or upload fails, the callback raises
`PersistentBackupError` before the next epoch. Local checkpoint/history bytes are
not deleted or rewritten.

### Explicit restore in a fresh Kaggle session

Attach the private checkpoint Dataset, keep
`RESTORE_PERSISTENT_CHECKPOINTS = False` until a deliberate restore decision,
then point the notebook at exactly one directory under `/kaggle/input`. The
`restore-persistent` command:

- starts no training;
- rejects TEST-like content and ambiguous checkpoint copies;
- copies only allowlisted files using temporary destinations;
- verifies SHA-256 and size and prints them for checkpoint/history files;
- refuses to overwrite a different existing artifact.

After copying, the normal history, provenance, policy, taxonomy, optimizer,
checkpoint, and TEST-lock validators remain authoritative. Restore never bypasses
or weakens the existing resume gates.

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

Keep both notebook authorization switches `False`, keep `INTERRUPTED_PHASE_ACTION = 'fail'`, keep persistent backup and restore disabled, and preserve recovery artifacts until separate human decisions. No INTERNAL TEST or PlantDoc TEST dataset is needed or permitted for this workflow.
