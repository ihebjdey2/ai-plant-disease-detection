# Dataset V2 logical cleaning policy

> Historical note: this document records the original Step 5C dHash screening
> decisions. Step 5C.1 subsequently corrected connected-component chaining with
> pHash, geometry, ORB, and direct representative grouping. See
> `docs/dataset-v2-perceptual-resolution.md`. The original reports and counts
> below remain intentionally preserved for audit transparency.

## Scope

Step 5C converts the audited Dataset V2 inventory into deterministic manifests. It does not edit, move, rename, resize, or delete source images. It also does not create a train/validation split, balance classes, augment data, or train a model.

The source inventory contains 21,124 records:

- 19,304 semantically matched training candidates;
- 1,584 training records with ambiguous or unsupported mappings;
- 236 locked PlantDoc TEST benchmark records.

The authoritative generated outputs are:

- `training/datasets/manifests/dataset-v2-master.csv`: every audited source record and its logical decision;
- `training/datasets/manifests/dataset-v2-clean-candidates.csv`: only records currently marked `INCLUDE`;
- `training/datasets/reports/dataset-v2-cleaning-summary.json`: deterministic totals and breakdowns;
- `training/datasets/reports/exact-label-conflicts.csv`: all members of exact conflicting-label groups;
- `training/datasets/reports/perceptual-group-members.csv`: compact record-to-group assignments;
- `training/datasets/reports/perceptual-groups-summary.csv`: one row per perceptual connected component;
- `training/datasets/reports/perceptual-review-queue.csv`: one representative row per high-risk group and risk category.

## Decision vocabulary

Every master record receives one status:

- `INCLUDE`: currently eligible for a future training split;
- `EXCLUDE`: deterministically ineligible;
- `REVIEW`: potentially eligible, but a human decision is required first.

Reasons use a controlled vocabulary rather than free text:

- `LOCKED_BENCHMARK`
- `EXACT_BENCHMARK_LEAKAGE`
- `EXACT_DUPLICATE_COPY`
- `EXACT_LABEL_CONFLICT`
- `PERCEPTUAL_LABEL_CONFLICT`
- `PERCEPTUAL_BENCHMARK_MATCH`
- `UNRESOLVED_MAPPING`
- `UNSUPPORTED_CLASS`
- `NOT_SEMANTIC_MATCH`

## Exact duplicates

An exact duplicate has the same SHA-256 digest and therefore the same bytes. For an exact group whose eligible records all have the same target class, the builder keeps one canonical record. Because all current candidates are verified originals, the canonical choice is the first record in deterministic dataset, source-version, source-path, and record-ID order. Other copies are excluded as `EXACT_DUPLICATE_COPY`. The images remain untouched on disk.

The audit contains 835 exact groups and 882 copies beyond one canonical content. After applying benchmark and label-conflict precedence, 863 records are excluded specifically as duplicate copies.

## Exact label conflicts

Identical bytes with different source labels or deployed targets are unsafe supervision. The builder does not guess which label is correct. All training members of a non-benchmark conflict group are excluded as `EXACT_LABEL_CONFLICT`.

There are 16 detected conflict groups. Eight overlap locked PlantDoc TEST groups and therefore follow the higher-priority benchmark policy. The other eight groups cause 16 training-record exclusions under `EXACT_LABEL_CONFLICT`.

## Locked benchmark protection

Every PlantDoc TEST row remains `EXCLUDE / LOCKED_BENCHMARK`. If a training record is byte-identical to the locked benchmark, the training record becomes `EXCLUDE / EXACT_BENCHMARK_LEAKAGE`.

The audit contains 11 exact TRAIN-to-TEST leakage groups and 11 training-side exclusions. PlantDoc TEST is never copied into the candidate pool.

## Perceptual similarity

dHash detects visually similar structure, but it does not prove that two photographs show the same observation. Similar leaves, uniform backgrounds, burst photography, crops, resizes, and rotations can all produce a low Hamming distance. Step 5C therefore does not blindly delete every pair at distance 4 or lower.

The 1,445,318 pair relations are converted into 2,351 connected components covering 8,066 records. This keeps the relationship needed by the future split without committing the 288 MB pairwise file.

Pairs are triaged as:

1. `TRAIN_TO_LOCKED_BENCHMARK` — highest priority;
2. `DIFFERENT_TARGET` — possible label conflict;
3. `SAME_TARGET_CROSS_SOURCE` — low risk;
4. `SAME_TARGET_SAME_SOURCE` — low risk.

Same-target records remain eligible unless another rule excludes them. Their shared `near_duplicate_group_id` ensures that a future split can keep related observations together.

Direct training endpoints of a different-target pair become `REVIEW / PERCEPTUAL_LABEL_CONFLICT`. Direct training endpoints paired with PlantDoc TEST become `REVIEW / PERCEPTUAL_BENCHMARK_MATCH`. dHash alone never produces an automatic exclusion.

## Compact manual-review queue

The raw perceptual audit contains 883,702 different-target pairs and four TRAIN-to-PlantDoc-TEST pairs. Reviewing every pair is impossible and mostly redundant. They collapse into:

- 75 different-target groups;
- 4 benchmark-match groups;
- 78 unique high-risk components because one component contains both risk categories;
- 79 representative review rows, one for each group/category combination.

The largest component contains 2,763 PLDD-UP records spanning Potato Early blight, Potato Late blight, and Potato healthy. It alone represents 1,441,693 pair relations, including 883,566 different-target pairs. This is a strong warning that low-resolution dHash is over-grouping this capture collection; a human must decide how that component should be handled before splitting.

## Current deterministic result

Within the 19,304 semantically matched training candidates:

| Decision | Records |
|---|---:|
| INCLUDE | 15,446 |
| REVIEW | 2,971 |
| EXCLUDE | 887 |

Deterministic exclusions inside that semantic pool consist of 863 exact duplicate copies, 16 records from non-benchmark exact label conflicts, and 8 semantically matched exact benchmark leaks. The full master also contains three ambiguous exact benchmark leaks, 1,254 other unresolved mappings, 327 unsupported mappings, and 236 locked TEST rows, producing 2,707 total `EXCLUDE` rows across all 21,124 records.

## Counts by source

| Source | Semantic candidates | INCLUDE | EXCLUDE | REVIEW | Unique SHA-256 contents |
|---|---:|---:|---:|---:|---:|
| PlantDoc TRAIN | 1,085 | 1,034 | 10 | 41 | 1,084 |
| Banu/Deb Potato originals | 36 | 24 | 12 | 0 | 24 |
| Seasonal Corn originals | 2,664 | 2,046 | 606 | 12 | 2,060 |
| PLDD-UP | 15,519 | 12,342 | 259 | 2,918 | 15,265 |

## Counts by deployed target

| Target class | Raw | INCLUDE | EXCLUDE | REVIEW | Sources | Near groups |
|---|---:|---:|---:|---:|---:|---:|
| Apple Apple scab | 83 | 83 | 0 | 0 | 1 | 1 |
| Corn Cercospora leaf spot | 1,561 | 950 | 606 | 5 | 2 | 63 |
| Corn Common rust | 129 | 129 | 0 | 0 | 1 | 2 |
| Corn healthy | 1,038 | 1,029 | 2 | 7 | 1 | 433 |
| Grape Black rot | 56 | 56 | 0 | 0 | 1 | 1 |
| Potato Early blight | 4,912 | 3,694 | 64 | 1,154 | 2 | 319 |
| Potato Late blight | 6,233 | 5,055 | 49 | 1,129 | 3 | 351 |
| Potato healthy | 4,616 | 3,802 | 164 | 650 | 2 | 1,196 |
| Squash Powdery mildew | 124 | 124 | 0 | 0 | 1 | 2 |
| Tomato Bacterial spot | 101 | 98 | 0 | 3 | 1 | 5 |
| Tomato Early blight | 79 | 69 | 0 | 10 | 1 | 12 |
| Tomato Late blight | 101 | 93 | 0 | 8 | 1 | 11 |
| Tomato Leaf Mold | 85 | 85 | 0 | 0 | 1 | 1 |
| Tomato Septoria leaf spot | 140 | 133 | 2 | 5 | 1 | 11 |
| Tomato Spider mites | 2 | 2 | 0 | 0 | 1 | 0 |
| Tomato mosaic virus | 44 | 44 | 0 | 0 | 1 | 1 |

These counts expose severe imbalance: Potato classes dominate, while Tomato Spider mites has only two eligible records. Step 5C reports this condition but does not undersample, oversample, duplicate, augment, or compute class weights.

## Rebuild and validation

Run from the repository root:

```powershell
python scripts/build_dataset_v2_manifest.py
```

When the ignored exhaustive dHash report is available, the command regenerates compact perceptual grouping reports. Without it, the command uses the committed group-membership and compact reports to reproduce the master and clean manifests without network access, raw images, or TensorFlow.

The builder fails if an `INCLUDE` row uses PlantDoc TEST, has an unresolved mapping, uses a target outside the deployed taxonomy, belongs to an exact label conflict, shares an included SHA-256 content, or appears in the clean manifest while still requiring review.

## Why no train/validation split exists yet

The 2,971 review records and 79 high-risk group/category decisions must be resolved first. Creating a split now could leak related photographs across partitions or preserve uncertain labels. The next split-design step must assign every exact or near-duplicate group as an indivisible unit.
