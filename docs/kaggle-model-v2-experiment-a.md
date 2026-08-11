# Model V2 Experiment A: isolated TensorFlow 2.15 on Kaggle

Kaggle currently exposes a useful free Tesla P100 GPU, but its notebook image may use Python 3.12, TensorFlow 2.20, Keras 3, and NumPy 2. Experiment A must not silently inherit that scientific stack. The approved baseline remains Python 3.11, TensorFlow 2.15.0, Keras 2.15.0, and NumPy 1.26.4.

The notebook [`notebooks/kaggle_model_v2_experiment_a.ipynb`](../notebooks/kaggle_model_v2_experiment_a.ipynb) therefore treats the Kaggle kernel as an orchestrator only. It creates an isolated Python 3.11 environment under `/kaggle/working` and launches every TensorFlow operation through an explicit subprocess. It does not uninstall, replace, or import TensorFlow 2.15 into the Python 3.12 kernel.

## Safety boundaries

- TRAIN: 58,857 images and 39/39 classes.
- VALIDATION: 7,362 images and 39/39 classes.
- INTERNAL TEST images: not attached or loaded.
- PlantDoc TEST: not attached or loaded.
- `START_TRAINING = False` by default.
- No class weights.
- Production `plant_disease_model.h5`, taxonomy, no-leaf behavior, and the 60% threshold remain unchanged.
- Runtime preparation and preflight never call `model.fit()`.

## Runtime layout

The bootstrap installs `uv==0.12.3` with Astral's official versioned standalone installer. It sets `UV_UNMANAGED_INSTALL` to the runtime-controlled `uv-bin` directory, so it does not need Kaggle host `pip`, `venv`, `ensurepip`, or `pipx`, and it does not modify shell profiles or `PATH`. uv then installs a managed Python 3.11 and creates the TensorFlow environment without touching system Python:

```text
/kaggle/working/agridiagnose-tf215-runtime/
├── uv-bin/
│   └── uv                                # pinned standalone executable
├── uv-installer.sh                       # versioned official installer
├── uv-python/                            # uv-managed CPython 3.11
├── uv-cache/
├── venvs/
│   └── agridiagnose-tf215/
│       └── bin/python                    # Experiment A interpreter
├── bootstrap.json
├── tf215-gpu-runtime.json
├── experiment-a-config.json
└── experiment-a-preflight.json
```

The isolated requirements are committed in `requirements-kaggle-tf215.txt`. They pin:

- `tensorflow[and-cuda]==2.15.0`;
- `keras==2.15.0`;
- `numpy==1.26.4`;
- only the image, metrics, tabular, plotting, and HDF5 dependencies needed by Experiment A.

The CUDA extra installs TensorFlow's Python-side NVIDIA dependencies inside the isolated environment. The Kaggle host driver remains untouched.

## Required private Kaggle datasets

Attach five private datasets that preserve the existing source-relative paths and contain only files required by TRAIN and VALIDATION:

| Configuration key | Expected input root |
|---|---|
| `historical` | `/kaggle/input/agridiagnose-historical` |
| `pldd_up` | `/kaggle/input/agridiagnose-pldd-up` |
| `seasonal_corn` | `/kaggle/input/agridiagnose-seasonal-corn` |
| `plantdoc_train` | `/kaggle/input/agridiagnose-plantdoc-train` |
| `banu_deb` | `/kaggle/input/agridiagnose-banu-deb` |

PlantDoc and Banu/Deb use committed `local_file` aliases. Preserve those filenames. Do not upload or attach INTERNAL TEST images or the official PlantDoc TEST split.

## Beginner workflow

1. Open Kaggle and create a Notebook.
2. In **Settings**, choose **Accelerator → GPU** and enable Internet.
3. Use **Add Input → Your Datasets** to attach the five private TRAIN/VALIDATION sources.
4. Import `notebooks/kaggle_model_v2_experiment_a.ipynb`.
5. Run the first cell. It reports the Kaggle system kernel and `nvidia-smi`. Python 3.12 / TensorFlow 2.20 here is acceptable because it is not used for Experiment A.
6. Run the repository cell. It clones the public repository at the immutable revision embedded in the notebook.
7. Run the bootstrap cell. It downloads the official `uv==0.12.3` standalone installer, creates managed Python 3.11 and the isolated venv, and installs the directly pinned TensorFlow 2.15 dependencies under `/kaggle/working`. The resulting complete package set is checked and recorded by the runtime report.
8. Run the isolated runtime gate. It must report all of the following:
   - Python `3.11.x`;
   - TensorFlow `2.15.x`;
   - Keras `2.15.x`;
   - NumPy `1.26.4`;
   - `tf.test.is_built_with_cuda() == True`;
   - at least one TensorFlow GPU;
   - matrix multiplication placed on `/GPU:0`.
9. If any condition fails, stop at `KAGGLE_TF215_GPU_RUNTIME_FAILED`. Do not train with system TensorFlow 2.20.
10. Review the folders printed from `/kaggle/input`, then edit only `SOURCE_ROOTS` if the slugs differ.
11. Keep `START_TRAINING = False` and write the execution configuration.
12. Run the isolated preflight cell. It verifies every TRAIN/VALIDATION file, MacroF1, preprocessing, model shapes, fresh ImageNet initialization, Phase 1 freezing, Phase 2 boundary, and BatchNormalization freeze policy.
13. Confirm TRAIN `58,857/58,857`, VALIDATION `7,362/7,362`, missing/corrupt `0`, and coverage `39/39`.
14. Download or copy `tf215-gpu-runtime.json` and `experiment-a-preflight.json` for review.
15. Stop. Do not enable the final training cell until separate human approval.

## Commands executed by the notebook

Bootstrap, launched by system Python in isolated mode with `PYTHONPATH`, `PYTHONHOME`, and `VIRTUAL_ENV` removed from the child environment:

```bash
python scripts/bootstrap_kaggle_tf215_runtime.py \
  --working-root /kaggle/working/agridiagnose-tf215-runtime \
  --project-root /kaggle/working/ai-plant-disease-detection
```

The notebook also sets `PYTHONNOUSERSITE=1` for bootstrap, runtime-gate, preflight, and future training subprocesses. The bootstrap removes an inherited `UV_INSTALL_DIR` (Kaggle may set it to `/usr/local/bin`) before enforcing `UV_UNMANAGED_INSTALL` under the runtime root. It preserves `PATH`, CUDA, NVIDIA, and driver-related environment variables. This prevents both host `sitecustomize` contamination and installer-path overrides without disabling GPU access.

Runtime gate, launched with isolated Python 3.11:

```bash
/kaggle/working/agridiagnose-tf215-runtime/venvs/agridiagnose-tf215/bin/python \
  scripts/run_kaggle_model_v2_experiment_a.py verify-runtime \
  --output /kaggle/working/agridiagnose-tf215-runtime/tf215-gpu-runtime.json
```

Preflight, also launched with isolated Python 3.11:

```bash
/kaggle/working/agridiagnose-tf215-runtime/venvs/agridiagnose-tf215/bin/python \
  scripts/run_kaggle_model_v2_experiment_a.py preflight \
  --config /kaggle/working/agridiagnose-tf215-runtime/experiment-a-config.json \
  --output /kaggle/working/agridiagnose-tf215-runtime/experiment-a-preflight.json
```

The future training command requires both `start_training: true` in the validated configuration and the explicit `--authorize-training` flag. Without both, it stops with `TRAINING_DISABLED_BY_USER`.

## Runtime verification details

The isolated verifier prints and records Python, TensorFlow, Keras, NumPy, CUDA build status, TensorFlow GPU devices, `nvidia-smi` information, and the actual device of a small matrix multiplication. TensorFlow soft device placement is disabled for the smoke test, preventing a silent CPU fallback.

The provided Kaggle system audit reported a Tesla P100-PCIE-16GB and driver `580.159.04`. That confirms the host GPU, not yet TensorFlow 2.15 compatibility. Only `tf215-gpu-runtime.json`, produced by the isolated interpreter, can validate the approved runtime.

## Preflight details

The isolated preflight:

- loads only the TRAIN and VALIDATION manifests;
- validates every referenced image with Pillow;
- requires zero missing and zero corrupt images;
- checks RGB `224×224×3`, float32, `[0,1]` preprocessing;
- confirms augmentation only on TRAIN;
- validates epoch-level MacroF1 against scikit-learn within `1e-6`;
- verifies the INTERNAL TEST manifest hash without loading its records or images;
- does not access PlantDoc TEST;
- downloads fresh official ImageNet MobileNetV2 weights;
- verifies input `(None,224,224,3)` and output `(None,39)`;
- verifies the entire backbone is frozen for Phase 1;
- verifies fine-tuning begins at `block_13_expand` and all 52 BatchNormalization layers remain frozen for Phase 2;
- records `training_performed: false`.

## Troubleshooting

### `KAGGLE_GPU_NOT_AVAILABLE`

Select **Settings → Accelerator → GPU**, restart the session, and rerun the system audit.

### Bootstrap or uv download failure

Confirm Internet is enabled, `curl` and `sh` are available, and `/kaggle/working` has sufficient free space. Rerunning the bootstrap is safe: the pinned standalone uv executable is reinstalled and version-checked; managed Python is reused; a missing, broken, or non-3.11 experiment venv is recreated with `uv venv --clear`; and the pinned requirements are installed and checked again. The obsolete partial `uv-bootstrap/` directory from the host-pip implementation is ignored and can no longer block recovery.

### `uv-bootstrap/bin/python: No module named pip`

This was the real failure of the previous bootstrap. The corrected workflow never invokes that Python or host `pip`. Rerun Step 3 with the notebook pinned to the corrected implementation revision; the official standalone installer writes `uv` directly to `runtime/uv-bin/uv`.

### Installer reports `installing to /usr/local/bin`

Kaggle may export `UV_INSTALL_DIR=/usr/local/bin`. Older bootstrap revisions allowed that host value to override the unmanaged runtime destination, then failed because `runtime/uv-bin/uv` did not exist. The corrected bootstrap removes the inherited value and enforces `UV_UNMANAGED_INSTALL=/kaggle/working/agridiagnose-tf215-runtime/uv-bin` before invoking the installer.

### `uv 0.12.3 (x86_64-unknown-linux-gnu)` version mismatch

Linux includes the target platform in `uv --version`. The validator accepts this official suffix but still requires the parsed semantic version to equal exactly `0.12.3`; different versions or arbitrary trailing text remain rejected.

### `KAGGLE_TF215_GPU_RUNTIME_FAILED`

Read `tf215-gpu-runtime.json`. Stop if Python/package versions, CUDA build, GPU enumeration, or the GPU smoke device fails. Never fall back to system TensorFlow 2.20 or CPU training.

### Missing datasets

Inspect the printed `/kaggle/input` entries and correct only the five source-root values. Do not train an incomplete subset.

### P100 memory error later

The approved initial batch size is 32. After a real OOM, 16 and then 8 are permitted fallbacks, but the reason must be recorded. Do not modify architecture, resolution, augmentation, or learning rates.

### Session interruption

Everything under `/kaggle/working` is session-scoped. Preserve reports and checkpoints through Kaggle outputs. Exact mid-epoch continuation is not claimed; restart the affected phase explicitly rather than inventing continuity.

## Stop condition

This preparation step ends after the isolated GPU gate, model audit, and complete TRAIN/VALIDATION preflight. Keep `START_TRAINING = False`. Do not evaluate either TEST set and do not create a candidate model yet.
