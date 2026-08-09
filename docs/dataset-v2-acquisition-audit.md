# Dataset V2 Step 5B and 5B.1 Acquisition Audit

Audit date: 2026-08-09

Scope: PlantDoc TRAIN, Banu/Deb Potato originals, Seasonal Corn originals, PLDD-UP, and read-only leakage comparison against locked PlantDoc TEST.

Status: acquisition and global audit completed; no cleaned dataset or split has been approved.

## Guardrails

- PlantDoc TEST remained locked and read-only. No TEST image was materialized under the training data root.
- No model training, fine-tuning, threshold tuning, augmentation, balancing, cleaning, or split creation was performed.
- Raw source paths and image bytes were preserved. Windows extended paths are used only to support long official filenames.
- Downloads, raw images, exhaustive local reports, and logs remain ignored by Git.
- PlantWild, PlantSeg, Kaggle mirrors, and other unapproved sources were not acquired.

## Storage

- Free disk before acquisition: 215.65 GiB
- Verified official archives: 12.46 GiB
- Verified extracted additions: 13.25 GiB
- Conservative temporary-space allowance used before download: 40 GiB
- Free disk after acquisition and removal of one invalid resumed transfer: 188.96 GiB

The acquisition stayed within the approved safety margin.

## Sources and integrity

### PlantDoc

- Official source: <https://github.com/pratikkayal/PlantDoc-Dataset>
- Immutable revision: `5467f6012d78d1c446145d5f582da6096f852ae8`
- TRAIN: 2,342 valid images, 0 corrupted
- Semantic matches: 1,085 across 13 deployed classes
- Ambiguous and excluded: 1,257
- TEST: 236 read-only images used only for leakage checks

Generic crop-leaf labels are not treated as healthy, and generic diseases are not forced into more specific deployed classes.

### Banu/Deb Potato Leaf Disease Dataset

- Official source: <https://doi.org/10.17632/d5b3fzpw3g.1>
- Official V1 archive SHA-256: `549c7f3343422fa2b77b6fb2c5009a52215aa00626b2646435ba19f4826f8192`
- Archive images: 2,351
- Verifiable `orig_*` images: 84
- Excluded `aug_*` images: 2,267
- Valid originals: 84; corrupted: 0
- Semantic matches: 36; NOT_SUPPORTED: 48

The published description and archive contents disagree on original counts. Only the 84 files explicitly named `orig_*` remain in the audit.

### Seasonal Corn originals

- Official source: <https://doi.org/10.17632/vy629dngm8.1>
- Version: 1
- Author: MD Hasan Ahmad
- License: CC BY 4.0
- Mendeley file ID: `e086f779-470a-4c7c-ba81-97734e9f8dd6`
- Archive SHA-256: `575628df92e69c169fa82c8506253d7d5886a8931605bf765f0e2577022dc479`
- Compressed size: 4,599,036,050 bytes
- Extracted image bytes: 4,813,231,041

The official page reports 2,943 originals and 7,500 generated augmentations. The actual V1 archive contains exactly the 2,943 reported original class counts and no augmented directory or additional augmented files. Originals are therefore reliably separable without appearance-based inference.

| Source label | Verified images | Mapping | Target |
|---|---:|---|---|
| Bacterial Leaf Streak | 190 | NOT_SUPPORTED | - |
| Common_rust | 129 | MATCHED | Corn Common rust |
| Gray_leaf_spot | 1,497 | MATCHED | Corn Cercospora leaf spot |
| Healthy | 1,038 | MATCHED | Corn healthy |
| Maize Chlorotic Mottle Virus | 89 | NOT_SUPPORTED | - |

Integrity results:

- Valid originals: 2,943
- Semantic matches: 2,664
- NOT_SUPPORTED: 279
- Ambiguous: 0
- Corrupted or unsupported files: 0
- Formats: 2,225 JPEG and 718 PNG
- Dimensions: 200 x 200 minimum components; up to 4,160 x 4,160

The source describes high-resolution captures from several corn farms in Gurudaspur, Natore, Rajshahi, Bangladesh, with environmental metadata and expert-assisted classes.

### PLDD-UP

- Official source: <https://doi.org/10.17632/3j4nfkvp2n.1>
- Version: 1
- Authors: Prakash Kumar Singh, Arun Yadav, Divakar Yadav, Sarthak Tiwari, and Aseem Chandel
- License: CC BY 4.0
- Archives: `EB.zip`, `Healthy.zip`, and `LB.zip`
- Combined compressed size: 8,772,692,847 bytes
- Combined extracted image bytes: 9,407,440,027 bytes

Verified archive hashes:

| Archive | Mendeley file ID | SHA-256 |
|---|---|---|
| EB.zip | `5717ac85-cf61-461d-bf70-e1e5af2f2c53` | `cffd37bbb79e75c0e23c1486f88f0a7c873b3fe67f643c41db3abd794bdc01e5` |
| Healthy.zip | `d4ce2acf-3af8-416e-90bc-bb834ba9da66` | `2b7e4107d7ba03c0ef9636831aa4de7d333435df238f667f562e27b1e46e59e2` |
| LB.zip | `35ff7712-865c-41bf-96b1-d25d84af7b95` | `f4d31182b5d2f147c256e1c73838eac9b17592b89ed2b1ae394f58863e74a447` |

The official metadata names the folders `EB`, `LB`, and `Healthy`. The official archive filenames explicitly use `early-blight` inside EB and `late-blight` inside LB, confirming their semantics without relying only on abbreviations.

| Source label | Verified images | Mapping | Target |
|---|---:|---|---|
| EB | 4,803 | MATCHED | Potato Early blight |
| Healthy | 4,600 | MATCHED | Potato healthy |
| LB | 6,116 | MATCHED | Potato Late blight |

Integrity results:

- Valid originals: 15,519
- Semantic matches: 15,519
- Ambiguous or NOT_SUPPORTED: 0
- Corrupted or unsupported files: 0
- Pillow formats: 15,374 JPEG and 145 files decoded as MPO despite JPEG source extensions
- Dimensions: width 116-8,160 pixels; height 198-4,624 pixels

The source documents operational potato fields in Mainpuri, Etawah, and Jaswantnagar, Uttar Pradesh, during the October 2025-March 2026 Rabi season. Captures used a Nikon D5300 and smartphones of at least 12 MP under natural daylight at different times of day.

## Candidate counts before deduplication

| Dataset | Exact semantic candidates |
|---|---:|
| PlantDoc TRAIN | 1,085 |
| Banu/Deb Potato originals | 36 |
| Seasonal Corn originals | 2,664 |
| PLDD-UP | 15,519 |
| Total | 19,304 |

| Deployed target | Candidates |
|---|---:|
| Apple Apple scab | 83 |
| Corn Cercospora leaf spot | 1,561 |
| Corn Common rust | 129 |
| Corn healthy | 1,038 |
| Grape Black rot | 56 |
| Potato Early blight | 4,912 |
| Potato Late blight | 6,233 |
| Potato healthy | 4,616 |
| Squash Powdery mildew | 124 |
| Tomato Bacterial spot | 101 |
| Tomato Early blight | 79 |
| Tomato Late blight | 101 |
| Tomato Leaf Mold | 85 |
| Tomato Septoria leaf spot | 140 |
| Tomato Spider mites | 2 |
| Tomato mosaic virus | 44 |

These are audit counts, not final training counts.

## Global duplicate audit

The global audit covers 21,124 records: all four candidate sources plus locked PlantDoc TEST.

### Exact SHA-256

- Exact duplicate groups: 835
- Duplicate images beyond the first member: 882
- Seasonal Corn groups: 604, representing 604 copies beyond the first
- PLDD-UP groups: 215, representing 254 copies beyond the first
- PlantDoc groups: 12
- Banu/Deb groups: 4
- Exact duplicate groups across different candidate datasets: 0
- Groups touching locked PlantDoc TEST: 11
- Training images byte-identical to locked TEST: 11

The 11 TEST leakage groups are the same PlantDoc cross-split findings already identified in Step 5B. Seasonal Corn and PLDD-UP introduce no new exact TEST leakage.

### Label conflicts

Sixteen exact groups carry different source labels or target classes:

- Existing PlantDoc conflicts: 9
- Seasonal Corn: 2 Gray leaf spot versus Healthy pairs
- PLDD-UP: 5 Early blight versus Healthy pairs

Every semantically matched member of an exact conflict is marked `EXCLUDE_FROM_TRAINING`. No label was chosen arbitrarily.

### Perceptual dHash screening

Using the unchanged 64-bit dHash threshold of 4:

- Candidate pairs: 1,445,318
- Pairs touching locked TEST: 5, including 4 TRAIN-to-TEST and 1 TEST-to-TEST
- Cross-dataset candidate pairs: 1
- New Seasonal Corn or PLDD-UP pairs touching PlantDoc TEST: 0

The single new cross-dataset pair links a Seasonal Corn Healthy image to a PLDD-UP Late blight image at distance 4. It is `NEEDS_MANUAL_REVIEW`; dHash alone is not treated as duplicate proof.

The exhaustive 288 MB pair list remains local and ignored. Its SHA-256 is `ab800ce548e6bb5bad9a2fd2d79b96b2d775dce97846284c7667a38f96a85b32`. Git stores all TEST and cross-dataset rows, deterministic group samples, and complete aggregate counts. Nothing was truncated silently.

## Candidate status after audit recommendations

| Dataset | APPROVED_CANDIDATE | EXCLUDE_FROM_TRAINING | NEEDS_MANUAL_REVIEW | Other |
|---|---:|---:|---:|---:|
| PlantDoc TRAIN | 992 | 10 | 83 | 1,257 AMBIGUOUS |
| Banu/Deb originals | 18 | 0 | 18 | 48 NOT_SUPPORTED |
| Seasonal Corn originals | 464 | 4 | 2,196 | 279 NOT_SUPPORTED |
| PLDD-UP | 8,151 | 10 | 7,358 | 0 |

These statuses are recommendations only. No image was removed, copied into a clean dataset, or assigned to train/validation.

## Reproducible artifacts

- `training/datasets/manifests/dataset-v2-global-audit.csv`
- `training/datasets/manifests/seasonal-corn-originals.csv`
- `training/datasets/manifests/pldd-up.csv`
- `training/datasets/mappings/seasonal-corn.json`
- `training/datasets/mappings/pldd-up.json`
- `training/datasets/sources/seasonal-corn.json`
- `training/datasets/sources/pldd-up.json`
- `training/datasets/reports/exact-duplicate-groups.json`
- `training/datasets/reports/perceptual-duplicate-candidates.csv`
- `training/datasets/reports/perceptual-duplicate-summary.csv`
- `training/datasets/reports/step5b-audit-summary.json`

The manifests contain only source-relative paths and metadata. They do not contain private absolute filesystem paths.
