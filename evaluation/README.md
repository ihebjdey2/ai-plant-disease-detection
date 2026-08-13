# Offline model evaluation

This directory documents reproducible, offline evaluation of the existing
`plant_disease_model.h5` artifact. Evaluation never retrains or fine-tunes the
model and never downloads data.

## Dataset requirements

Use a genuinely held-out, labeled **test** dataset. Do not present performance
on the training set as generalization accuracy. If images may be duplicated
between train and test, audit and remove that leakage before interpreting the
metrics.

The dataset root must contain one directory per class. Supported images are
JPG, JPEG, PNG, and WEBP. The evaluator accepts only the following explicit
directory names, in the deployed model's semantic order:

```text
Apple___Apple_scab
Apple___Black_rot
Apple___Cedar_apple_rust
Apple___healthy
Background_without_leaves
Blueberry___healthy
Cherry___Powdery_mildew
Cherry___healthy
Corn___Cercospora_leaf_spot
Corn___Common_rust
Corn___Northern_Leaf_Blight
Corn___healthy
Grape___Black_rot
Grape___Esca
Grape___Leaf_blight
Grape___healthy
Orange___Huanglongbing
Peach___Bacterial_spot
Peach___healthy
Bell_pepper___Bacterial_spot
Bell_pepper___healthy
Potato___Early_blight
Potato___Late_blight
Potato___healthy
Raspberry___healthy
Soybean___healthy
Squash___Powdery_mildew
Strawberry___Leaf_scorch
Strawberry___healthy
Tomato___Bacterial_spot
Tomato___Early_blight
Tomato___Late_blight
Tomato___Leaf_Mold
Tomato___Septoria_leaf_spot
Tomato___Spider_mites
Tomato___Target_Spot
Tomato___Yellow_Leaf_Curl_Virus
Tomato___mosaic_virus
Tomato___healthy
```

Names from other PlantVillage distributions, such as
`Pepper__bell___healthy`, are not guessed or silently aliased. Rename a class
directory only after verifying that it has the same meaning as the deployed
class. Unknown directories make the evaluation fail clearly.

Place local data under `evaluation/dataset/` or `evaluation/datasets/`; both
locations are ignored by Git.

## Run

From the repository root:

```bash
python scripts/evaluate_model.py --dataset "PATH_TO_LABELED_TEST_DATASET"
```

Optional arguments:

```bash
python scripts/evaluate_model.py \
  --dataset "PATH_TO_LABELED_TEST_DATASET" \
  --model plant_disease_model.h5 \
  --batch-size 32 \
  --output-dir evaluation/results
```

Full evaluation requires all 39 classes. `--allow-subset` exists only for an
explicitly labeled partial analysis; missing classes remain visible and the
39-class confusion matrix is preserved.

For a subset, macro and weighted metrics are calculated over the classes that
actually have ground-truth support. The raw confusion matrix still preserves
all 39 deployed outputs, including predictions outside the evaluated subset.
A missing background class is reported as not evaluated rather than as an
artificial all-zero result.

## External PlantDoc subset benchmark

The conservative PlantDoc mapping is stored in
`evaluation/mappings/plantdoc.json`. It documents a review decision and reason
for every official PlantDoc TEST label. Generic or broader labels are excluded
instead of being silently treated as deployed healthy or disease classes.

PlantDoc includes file names that are invalid on Windows. The preparation
adapter can read the official Git tree directly, so the source repository is
not renamed or modified:

```powershell
python scripts/prepare_plantdoc_evaluation.py `
  --source evaluation/datasets/plantdoc `
  --output evaluation/datasets/plantdoc_prepared `
  --report evaluation/results/plantdoc/preparation.json `
  --deduplicate-exact

python scripts/evaluate_model.py `
  --dataset evaluation/datasets/plantdoc_prepared `
  --allow-subset `
  --output-dir evaluation/results/plantdoc
```

The preparation report records the original class counts, image integrity,
mapping exclusions, SHA-256 exact duplicates, perceptual near-duplicates, and
any explicit exact-deduplication. PlantDoc data, prepared images, and raw
benchmark results remain ignored by Git.

## Preprocessing and audit

Evaluation matches deployed inference exactly:

- convert to RGB;
- resize to 224 × 224;
- convert to `float32`;
- divide values by `255.0`;
- no augmentation and no MobileNetV2 `preprocess_input`.

Before model loading, the script reports class counts, missing and unexpected
classes, minimum/maximum samples, and corrupted images. Corrupted files are
skipped and listed. Evaluation stops if corruption leaves a represented class
empty.

## Outputs

The default output directory is `evaluation/results/`:

- `metrics.json`: accuracy, macro/weighted metrics, per-class metrics, dataset
  audit, model metadata, confidence analysis, and inference timing;
- `confusion_matrix.csv`: raw 39 × 39 matrix;
- `confusion_matrix.png`: readable plotted matrix with all class labels.

Generated outputs are ignored by default to prevent accidental publication of
private paths or unreviewed results. Copy only deliberately reviewed, small
portfolio artifacts to version control.

Accuracy measures the fraction of correct top-1 predictions. Precision asks
how often predictions of a class are correct; recall asks how many real samples
of that class are found; F1 balances precision and recall. Macro averages give
each model class equal weight, while weighted averages account for support.

The 60% confidence analysis is application-level reporting, not a 40th neural
network class. `Background without leaves` remains a normal class in the raw
39-class metrics and is also reported separately.

## Interpretation warning

PlantVillage-style images often have clean backgrounds and controlled capture
conditions. Strong held-out PlantVillage metrics do not prove equivalent field
performance under different lighting, cameras, cultivars, disease stages, or
backgrounds. This model provides decision support and is not an
agricultural-grade diagnosis system.

## Model V2 final evaluation

The evaluator and preparation documentation above remains valid for the legacy
deployed `plant_disease_model.h5` workflow. Model V2 now also has a frozen final
record covering candidate selection, a 7,344-image/39-class Dataset V2 INTERNAL
TEST, and an external PlantDoc TEST evaluation. The selected Model V2 artifact
has not yet replaced the deployed model.

The PlantDoc result is a conservative partial-class benchmark: 99 images across
12 semantically matched classes, not a full 39-class accuracy claim. Its much
lower external performance reveals substantial domain shift relative to the
internal Dataset V2 result.

See [Model V2 final evaluation and provenance](../docs/model-v2-final-evaluation.md)
for the frozen metrics, checksums, selection chronology, limitations, and
no-post-TEST-tuning policy.
