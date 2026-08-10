# Model V2 Experiment A on a free Kaggle GPU

This workflow prepares the approved, unweighted Experiment A without requiring paid cloud infrastructure. Training is performed manually in a Kaggle Notebook with a free NVIDIA GPU when Kaggle capacity is available. Nothing in this workflow changes the Flask application, the production `plant_disease_model.h5`, the 60% threshold, or the 39-class taxonomy.

The notebook is [`notebooks/kaggle_model_v2_experiment_a.ipynb`](../notebooks/kaggle_model_v2_experiment_a.ipynb).

## Safety boundaries

Experiment A uses only:

- 58,857 TRAIN images;
- 7,362 VALIDATION images;
- the approved `pixel / 255.0` preprocessing;
- TRAIN-only augmentation;
- no class weights.

It does not require or evaluate INTERNAL TEST images or PlantDoc TEST images. Prefer not to attach either locked dataset to the Kaggle notebook. The INTERNAL TEST manifest is read only as bytes to confirm its approved SHA-256; its records and images are not loaded.

## Required private Kaggle datasets

Create or upload five private Kaggle datasets while preserving the source layouts used by the existing manifests:

| Notebook key | Suggested Kaggle input | Source |
|---|---|---|
| `historical` | `/kaggle/input/agridiagnose-historical` | Historical Mendeley 39-class source |
| `pldd_up` | `/kaggle/input/agridiagnose-pldd-up` | PLDD-UP |
| `seasonal_corn` | `/kaggle/input/agridiagnose-seasonal-corn` | Seasonal Corn Leaf Disease Dataset |
| `plantdoc_train` | `/kaggle/input/agridiagnose-plantdoc-train` | PlantDoc TRAIN-source audit copy only |
| `banu_deb` | `/kaggle/input/agridiagnose-banu-deb` | Banu/Deb Potato audit copy |

PlantDoc and Banu/Deb use the committed `local_file` aliases in their provenance manifests. Their private Kaggle datasets must preserve those local audit-copy filenames. Do not rename source images or edit committed manifests.

## Beginner workflow

1. Sign in to Kaggle and select **Create → New Notebook**.
2. Open **Notebook options / Settings** and set **Accelerator → GPU**. Free GPU availability is controlled by Kaggle and may be temporarily unavailable.
3. Attach the five private TRAIN-source datasets listed above. Do not attach INTERNAL TEST or PlantDoc TEST.
4. Import `notebooks/kaggle_model_v2_experiment_a.ipynb`, or upload it as a notebook.
5. Keep Internet enabled so the public GitHub repository and official ImageNet MobileNetV2 weights can be downloaded.
6. Run the first runtime-audit cell. It prints Python, TensorFlow, Keras, NumPy, OS, CUDA status, TensorFlow devices, and `nvidia-smi`.
7. Run the scientific-stack gate. Do not install packages before inspecting these versions.
8. If the Python runtime is compatible but the approved versions are missing, explicitly enable the optional pinned-install cell, run it once, restart the Kaggle session, then rerun both gates. Never continue with TensorFlow 2.18+ or Keras 3 without a separately approved experiment change.
9. Run the repository cell. It clones the public repository and checks out the exact approved preparation revision recorded in the notebook.
10. Edit only the five entries in `SOURCE_ROOTS` if Kaggle assigned different input slugs.
11. Run the exhaustive preflight. Continue only when TRAIN is `58,857/58,857`, VALIDATION is `7,362/7,362`, missing/corrupt counts are zero, coverage is `39/39`, and MacroF1 passes.
12. Review the GPU memory information. Keep `BATCH_SIZE = 32` unless an actual out-of-memory error requires 16 or 8. Record any fallback.
13. Run the model-build and single-batch inference cell. This does not optimize weights.
14. Review every result, then change `START_TRAINING = True`.
15. Run Phase 1 and wait for completion. The full backbone is frozen, Adam uses `1e-3`, and the phase is limited to 10 epochs.
16. Run Phase 2 and wait for completion. Fine-tuning begins at `block_13_expand`, all BatchNormalization layers remain frozen, a new Adam optimizer uses `2e-5`, and the phase is limited to 20 epochs.
17. Run VALIDATION-only selection and artifact generation.
18. Open the Kaggle **Output** panel and download `/kaggle/working/agridiagnose-exp-a-results.zip`.
19. Preserve the ZIP and its SHA-256 unchanged for review. Do not replace the production model.

## Persistence and interrupted sessions

Best checkpoints are written immediately under:

`/kaggle/working/models/candidates/agri-diagnose-v2-exp-a/`

Each completed epoch atomically updates the relevant history CSV in:

`/kaggle/working/agridiagnose-exp-a-results/`

The notebook detects existing `phase1-*` and `phase2-*` artifacts. Exact mid-phase continuation is not claimed because callback and data-iterator continuity cannot be guaranteed across a free-session interruption. By default the notebook stops. Download existing artifacts first; then set `RESTART_INTERRUPTED_PHASE = True` only when intentionally restarting the affected phase. The resulting metadata records `fresh` or `restarted`.

Kaggle's `/kaggle/working` storage is session-scoped. Saving frequently reduces loss during a running session but is not a substitute for creating a notebook version or downloading outputs.

## Output archive

The final ZIP is designed to contain:

- `agri-diagnose-v2-exp-a.keras`;
- `environment.json`;
- `experiment.json`;
- `preflight.json`;
- `phase1-history.csv` and `phase2-history.csv`;
- `validation-metrics.json`;
- `validation-confusion-matrix.csv` and PNG;
- loss, accuracy, and MacroF1 learning curves;
- `model-v2-exp-a-summary.json`;
- a short experiment report.

Candidate selection uses VALIDATION only, in this order: highest MacroF1, lower loss, higher macro recall, then earlier epoch. The real-world VALIDATION slice contains 1,816 images across its supported classes.

## Troubleshooting

### `KAGGLE_GPU_NOT_AVAILABLE`

Open **Settings → Accelerator → GPU**, restart the session, and rerun the first cell. If Kaggle has no free capacity, save the notebook and try later. Never fall back to CPU silently.

### `KAGGLE_TF215_RUNTIME_INCOMPATIBLE`

The current Kaggle Python version cannot safely run TensorFlow 2.15. Stop. Do not change TensorFlow/Keras versions or redesign the experiment without human approval.

### `KAGGLE_APPROVED_STACK_REQUIRED`

Python is compatible but the active scientific packages do not match the approved stack. Review the displayed versions. If appropriate, manually enable the pinned-install cell, restart, and rerun the runtime and GPU gates.

### Missing datasets or images

Confirm all five private datasets are attached and edit only `SOURCE_ROOTS`. Preserve the source directory structure and PlantDoc/Banu local audit-copy names. A count below the approved total is a hard stop; do not train on a subset.

### Out of memory

Record the GPU model and free VRAM. Retry with batch size 16, then 8 only if a real OOM occurred. Do not change image size, augmentation, architecture, or optimizer policy.

### Session interruption

Download any available checkpoint/history artifacts. Restart only the affected phase using the explicit restart flag; do not fabricate missing epochs or combine incompatible histories. If `/kaggle/working` was lost, restart the experiment phase from its defined starting point.

## After the download

Do not evaluate INTERNAL TEST, rerun PlantDoc, start Experiment B, modify the threshold, or deploy the candidate. The ZIP must first be reviewed for environment integrity, histories, VALIDATION metrics, confusion patterns, and candidate hash.
