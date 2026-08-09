# Dataset V2 Step 5B Acquisition Audit

Audit date: 2026-08-09

Scope: PlantDoc TRAIN and the original-image subset in the Banu/Deb Potato Leaf Disease Dataset.

Status: acquisition and audit completed; no training dataset or split has been approved.

## Guardrails

- PlantDoc TEST remained locked and read-only. It was used only for leakage checks.
- No PlantDoc TEST image was copied into `training/datasets/raw/`.
- No model training, fine-tuning, threshold tuning, augmentation, balancing, or split creation was performed.
- Source archives and Git objects were not renamed or modified.
- Downloaded and materialized images remain ignored by Git.
- Seasonal Corn, PLDD-UP, PlantWild, PlantSeg, and sources with unclear terms were not acquired in this step.

## Sources and integrity

### PlantDoc

- Official source: <https://github.com/pratikkayal/PlantDoc-Dataset>
- Audited immutable revision: `5467f6012d78d1c446145d5f582da6096f852ae8`
- TRAIN paths: 2,342 across 28 source labels
- Valid images: 2,342
- Corrupted images: 0
- Formats decoded by Pillow: 2,335 JPEG, 5 PNG, and 2 JPEG files decoded as MPO
- TEST paths checked without materialization: 236 across 27 source labels
- Source repository change after audit: none

The conservative test mapping is reused from `evaluation/mappings/plantdoc.json`. TRAIN has one additional label, `Tomato two spotted spider mites leaf`, mapped exactly to `Tomato Spider mites` in `training/datasets/mappings/plantdoc-train-extra.json`.

The semantic review accepts 1,085 TRAIN images across 13 deployed classes and excludes 1,257 ambiguous images. Generic crop-leaf labels are not treated as healthy, and generic disease labels are not forced into more specific deployed diseases.

### Potato Leaf Disease Dataset

- Official source: <https://doi.org/10.17632/d5b3fzpw3g.1>
- Official v1 archive SHA-256: `549c7f3343422fa2b77b6fb2c5009a52215aa00626b2646435ba19f4826f8192`
- Archive entries: 2,351 JPG images
- Entries named `orig_*`: 84
- Entries named `aug_*`: 2,267
- Unsafe paths: 0
- Corrupted originals: 0

The archive contents do not match the landing-page description of 804 originals augmented to 2,400 images. The audit therefore makes no assumption that the missing originals exist. Only the 84 verifiable `orig_*` files were materialized; every `aug_*` file was excluded.

| Source label | Original files | Mapping | AgriDiagnose target |
|---|---:|---|---|
| Bacterial Soft Rot | 7 | NOT_SUPPORTED | — |
| Fungal Late Blight | 20 | MATCHED | Potato Late blight |
| Healthy | 16 | MATCHED | Potato healthy |
| Viral Leaf Roll | 33 | NOT_SUPPORTED | — |
| Viral PVX | 6 | NOT_SUPPORTED | — |
| Viral PVY | 2 | NOT_SUPPORTED | — |

This yields 36 semantically matched files and 48 excluded files before duplicate handling. The 16 Healthy files collapse to four unique SHA-256 values, so the matched Potato subset contains only 24 unique image contents before perceptual review.

## Duplicate audit

The audit covers 2,342 PlantDoc TRAIN candidates, 84 Potato originals, and 236 locked PlantDoc TEST images. Nothing was deleted automatically.

### Exact SHA-256 duplicates

- Duplicate groups: 16
- Duplicate images beyond the first member of each group: 24
- Groups spanning training candidates and locked TEST: 11
- Groups contained within training candidates: 5

The 11 cross-role groups each contain one PlantDoc TRAIN image and one byte-identical PlantDoc TEST image. Several also disagree on the source disease label, including Corn gray leaf spot versus leaf blight and Potato early blight versus late blight. These TRAIN members must be excluded from any future training pool before PlantDoc TEST can remain a valid external benchmark.

The five training-only groups consist of:

- four Potato Healthy groups, each containing four byte-identical files;
- one PlantDoc pair labeled once as Potato early blight and once as Potato late blight.

The conflicting PlantDoc pair requires manual label review. The Potato copies must not be counted as independent observations.

### Perceptual candidates

Using 64-bit difference hashes with a maximum Hamming distance of 4:

- Candidate pairs: 67
- PlantDoc TRAIN-to-TRAIN: 60
- PlantDoc TRAIN-to-locked-TEST: 4
- Potato original-to-original: 2
- PlantDoc locked-TEST-to-locked-TEST: 1

Exact SHA-256 matches are excluded from the perceptual list. The four non-identical TRAIN-to-TEST candidates require human review before any training selection. Perceptual hashes are screening signals, not proof that two images depict the same leaf.

## Current decision

The acquisition produced 1,121 semantically matched file records: 1,085 from PlantDoc TRAIN and 36 from Potato originals. This is not the final usable count. Exact TEST leakage, repeated Potato contents, ambiguous/conflicting labels, and perceptual candidates must be resolved before a training manifest or split is created.

PlantDoc TEST remains locked. No contaminated image has been approved for training, and no model result is claimed.

## Reproducible artifacts

- `training/datasets/manifests/plantdoc-train.csv`
- `training/datasets/manifests/potato-banu-deb-originals.csv`
- `training/datasets/mappings/plantdoc-train-extra.json`
- `training/datasets/mappings/potato-banu-deb.json`
- `training/datasets/reports/exact-duplicate-groups.json`
- `training/datasets/reports/perceptual-duplicate-candidates.csv`
- `training/datasets/reports/step5b-audit-summary.json`

The manifests contain source-relative paths, hashes, dimensions, formats, mapping status, and deployed targets. They contain no absolute local filesystem paths.
