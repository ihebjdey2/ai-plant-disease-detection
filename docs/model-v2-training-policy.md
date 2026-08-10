# Model V2 training policy

## Scope and holdout contract

Step 5G defines the reproducible policy and loader infrastructure for the first AgriDiagnose Model V2 experiments. It does not train, fine-tune, save, or evaluate a neural network.

Only these manifests may participate in development:

- `dataset-v2-train.csv` for optimization;
- `dataset-v2-validation.csv` for early stopping, checkpoint selection, experiment comparison, and future calibration decisions.

`dataset-v2-test.csv` is the frozen final internal test. PlantDoc TEST is a separate frozen external benchmark. Neither may be loaded during training-policy development or Experiment A/B selection. Code-level guards reject TEST records in development loaders.

## Historical Model V1 findings

The findings below come from `model.ipynb` and read-only inspection of `plant_disease_model.h5`. Unrecorded details remain `UNKNOWN`.

| Item | Verified historical behavior |
|---|---|
| Backbone | MobileNetV2 1.0, ImageNet initialization, `include_top=False`, average pooling |
| Input | RGB 224×224, batch size 32 |
| Preprocessing | `rescale=1/255` |
| Labels/loss | categorical one-hot labels; categorical cross-entropy |
| Head | Dense(128, ReLU) → Dropout(0.5) → Dense(128, ReLU) → Dropout(0.5) → Dense(39, softmax) |
| Phase 1 | backbone frozen; maximum 30 epochs; Adam configured by name |
| Phase 1 learning rate | `UNKNOWN` as an explicit project decision; the notebook relied on the Keras Adam default |
| Intended fine-tuning | first 100 backbone layers kept frozen; Adam 1e-5; maximum 10 epochs |
| Saved H5 state | 2,443,495 parameters; backbone parameter weights remain non-trainable; Adam 1e-5 stored in training config |
| Callbacks | EarlyStopping on `val_loss`, patience 5, restore best weights; best H5 checkpoint on `val_loss` |
| Validation metric | accuracy only |
| Class weights | none documented |

The notebook uses one `ImageDataGenerator` for both training and validation. Consequently rotation up to 40°, shifts/shear/zoom up to 20%, and horizontal flip also affect validation. This makes historical validation stochastic and is not retained in Model V2.

The notebook's fine-tuning cell changes selected inner-layer flags but does not explicitly set the frozen backbone model back to trainable. The saved H5 contains no trainable backbone parameter weights. Therefore successful V1 backbone fine-tuning is not established; the safest conclusion is that it likely remained effectively frozen.

## Preprocessing compatibility

Model V2 preserves deployed inference semantics:

`load → RGB → resize 224×224 → float32 → divide by 255.0`

The numeric range remains `[0,1]`. `mobilenet_v2.preprocess_input` is deliberately excluded because it would silently change the input range to approximately `[-1,1]` and make Model V2 incompatible with the existing inference contract unless the application were separately versioned.

## Dataset and imbalance

The policy reads class weights exclusively from 58,857 TRAIN records. VALIDATION contributes no counts to weighting.

- controlled historical TRAIN: 44,340;
- real-world TRAIN: 14,517;
- smallest class: Apple Cedar apple rust, 220;
- largest class: Potato Late blight, 5,685;
- mean class count: 1,509.15384615;
- median class count: 1,107;
- maximum/minimum ratio: 25.84090909.

The baseline preserves natural TRAIN membership: no physical oversampling, record duplication, majority-class undersampling, or domain weighting is permitted. Real-world samples are never downweighted merely because their capture conditions are less controlled.

## Moderate class weighting

Three candidates are recorded before training:

1. no weighting: `w_c = 1`;
2. inverse square-root count: `w_c ∝ sqrt(N / n_c)`;
3. median-frequency square-root: `w_c = sqrt(median(n) / n_c)`.

The two square-root formulas differ only by a constant before normalization. The recommended Experiment B weights use the median-frequency expression, then normalize so the sample-weighted mean is 1.0 and clip to `[0.5,3.0]`. On the current TRAIN distribution, normalized weights already fall between 0.55879069 and 2.84055404, so clipping does not alter them. The clipping policy remains an explicit safeguard for future audited manifests.

Full inverse-frequency weighting is rejected for the baseline because a 25.84× count ratio would transfer too much optimization influence to very small classes. Experiment A remains unweighted so the effect of moderate weights can be isolated using VALIDATION only.

## TRAIN-only augmentation

Dynamic augmentation is applied after deterministic RGB decoding, resize, and scaling, and only when `training=True` on TRAIN:

- horizontal and vertical flip;
- rotation up to ±15°;
- horizontal/vertical translation up to 10%;
- zoom approximately 0.9–1.1;
- brightness factor 0.1;
- contrast factor 0.1.

These transformations allow plausible leaf orientation and moderate capture variability while preserving lesion texture and morphology. VALIDATION, INTERNAL TEST, and PlantDoc TEST receive deterministic preprocessing only.

The baseline explicitly excludes MixUp, CutMix, synthetic generation, aggressive hue/saturation changes, shear transformations, strong perspective distortion, large crops that remove leaf evidence, and background replacement. They change semantics or introduce additional experimental variables before a trustworthy baseline exists.

## Model architecture

The baseline isolates the effect of Dataset V2 while preserving the existing model family:

`MobileNetV2(weights="imagenet", include_top=False) → GlobalAveragePooling2D → Dropout(0.25) → Dense(39, softmax)`

It starts cleanly from ImageNet rather than `plant_disease_model.h5`. This avoids inheriting the V1 classifier head and possible controlled-domain biases, making Dataset V2 and policy effects easier to attribute. The smaller head also avoids the unnecessary two-layer Dense stack used historically.

Sparse integer labels (`0..38`) are used with `SparseCategoricalCrossentropy`.

## Two-phase transfer learning

### Phase 1 — feature extraction

- freeze the entire MobileNetV2 backbone;
- train only the new classifier head;
- Adam learning rate: `1e-3`;
- maximum epochs: 10;
- early stopping may finish earlier.

### Phase 2 — upper-backbone fine-tuning

- make the backbone trainable, then keep layers before `block_13_expand` frozen;
- for TensorFlow 2.15 MobileNetV2, `block_13_expand` is layer index 116 of 154;
- the final 38 backbone layers are considered, but all BatchNormalization layers remain frozen;
- resulting policy: 25 non-BatchNorm upper layers trainable, 129 backbone layers frozen;
- Adam learning rate: `2e-5`;
- maximum epochs: 20.

Freezing all 52 MobileNetV2 BatchNormalization layers protects pretrained running statistics from destabilization under batch size 32 and mixed-domain batches. The layer name is authoritative; the recorded index is a TensorFlow 2.15 reproducibility check.

## Callbacks and selection

- EarlyStopping: monitor `val_loss`, patience 5, restore best weights.
- ReduceLROnPlateau: monitor `val_loss`, factor 0.2, patience 2, minimum learning rate `1e-7`.
- ModelCheckpoint: save the best development candidate by `val_macro_f1`, mode `max`.

The primary selection metric is overall VALIDATION macro F1. Secondary metrics are validation loss, accuracy, weighted F1, macro recall, and per-class precision/recall/F1. Macro F1 is primary because accuracy can conceal degradation in rare classes.

The validation metrics utility also reports a separate REAL_WORLD slice using only validation classes with actual real-world support. It does not fabricate zero-support metrics and does not replace overall validation. Five classes currently have robust real-world holdouts, ten limited holdouts, and Tomato Spider mites no real-world validation/test holdout, so no all-class real-world-generalization claim is permitted.

## Controlled experiment sequence

- Experiment A — `agri-diagnose-v2-exp-a`: the architecture and moderate augmentation above, no class weights.
- Experiment B — `agri-diagnose-v2-exp-b`: identical architecture, seed, augmentation, optimizer, phases, and epoch budgets, changing only to the recommended moderate class weights.

Only VALIDATION compares A and B. INTERNAL TEST and PlantDoc TEST remain unseen until the final candidate and all development decisions are frozen. Augmentation and class weighting are not tuned simultaneously.

## Reproducibility and data roots

The experiment seed is `20260810` for future Python, NumPy, and TensorFlow setup. Batch size defaults to 32 and may be lowered for GPU memory without changing dataset membership.

Manifest paths remain relative. Local roots are configured outside Git through:

- `AGRIDIAGNOSE_HISTORICAL_ROOT`;
- `AGRIDIAGNOSE_PLDD_UP_ROOT`;
- `AGRIDIAGNOSE_SEASONAL_CORN_ROOT`;
- `AGRIDIAGNOSE_PLANTDOC_TRAIN_ROOT`;
- `AGRIDIAGNOSE_BANU_DEB_ROOT`.

The resolver rejects absolute manifest paths and path traversal outside the configured root.

Run the non-training validation with:

```powershell
python scripts/prepare_model_v2_training.py --dry-run
```

Without configured roots, the command validates policies, manifests, taxonomy, counts, class weights, and holdout guards, then explicitly skips image inspection. With a local historical root configured, the Step 5G dry-run inspected eight deterministic source images and confirmed `(224,224,3)`, `float32`, `[0,1]`, and labels `0..38`.

## Versioning and next gate

Future checkpoints belong under `models/candidates/` with explicit experiment names. They must never overwrite `plant_disease_model.h5`. No candidate file is created in Step 5G.

The next human-approved step may run Experiment A, then Experiment B. It must implement the planned validation macro-F1 checkpoint integration, record full environment/version metadata, and still keep INTERNAL TEST and PlantDoc TEST locked during selection. The confidence threshold remains 60%, and class index 4 remains `Background without leaves` with unchanged `no_leaf` application behavior.
