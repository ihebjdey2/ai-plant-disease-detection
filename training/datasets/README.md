# Dataset V2 acquisition metadata

This directory stores versioned mappings, manifests, and audit reports. Dataset images, downloads, and local audit logs are intentionally excluded from Git.

## Local layout

```text
training/datasets/
├── downloads/       # ignored source archives
├── raw/             # ignored read-only materializations
├── local-audits/    # ignored execution logs
├── mappings/        # reviewed source-to-deployed taxonomy mappings
├── manifests/       # versioned per-image metadata and hashes
└── reports/         # versioned duplicate and integrity reports
```

PlantDoc TEST belongs under `evaluation/datasets/` and is a locked benchmark. It must never be copied into `training/datasets/raw/` or used for model selection.

## Audited inputs

1. Clone the official PlantDoc repository without checking out Windows-invalid filenames:

   ```powershell
   git clone --no-checkout https://github.com/pratikkayal/PlantDoc-Dataset evaluation/datasets/plantdoc
   ```

   The audit reads the immutable Git tree at `5467f6012d78d1c446145d5f582da6096f852ae8`, even if the clone's current `HEAD` later changes.

2. Download version 1 of the official Potato archive from <https://doi.org/10.17632/d5b3fzpw3g.1> as:

   ```text
   training/datasets/downloads/potato-leaf-disease-d5b3fzpw3g-v1.zip
   ```

   Expected SHA-256:

   ```text
   549c7f3343422fa2b77b6fb2c5009a52215aa00626b2646435ba19f4826f8192
   ```

3. Download the official Seasonal Corn V1 archive from <https://doi.org/10.17632/vy629dngm8.1> as:

   ```text
   training/datasets/downloads/seasonal-corn-v1.zip
   ```

   Expected SHA-256: `575628df92e69c169fa82c8506253d7d5886a8931605bf765f0e2577022dc479`.

4. Download the three official PLDD-UP V1 archives from <https://doi.org/10.17632/3j4nfkvp2n.1> as:

   ```text
   training/datasets/downloads/pldd-up-v1-eb.zip
   training/datasets/downloads/pldd-up-v1-healthy.zip
   training/datasets/downloads/pldd-up-v1-lb.zip
   ```

   Their expected hashes and Mendeley file identifiers are recorded in `training/datasets/sources/pldd-up.json`.

5. Download only the official non-augmented historical V1 archive from
   <https://doi.org/10.17632/tywbtsjrjv.1> as:

   ```text
   training/datasets/downloads/Plant_leaf_diseases_dataset_without_augmentation.zip
   ```

   Expected file ID: `d5652a28-c1d8-4b76-97f3-72fb80f94efc`.
   Expected SHA-256:
   `ac3432453984d02a86197987e775a5429d0d59e7cc7c35bcf5a8f50349b90ff0`.
   Extract it without changing source names under
   `training/datasets/raw/historical-mendeley-39/`. Do not acquire the
   pre-generated augmented archive for the historical base candidate pool.

## Run the audit

```powershell
python scripts/audit_dataset_v2_sources.py
```

The command validates all images, materializes PlantDoc TRAIN, only Banu/Deb Potato `orig_*` entries, Seasonal Corn originals, and PLDD-UP, verifies mappings, and regenerates the global duplicate reports. It never removes duplicates or creates train/validation/test splits.

The exhaustive dHash pair report is intentionally local because field-image bursts produce more than one million candidate pairs. Its SHA-256 and complete count are versioned in `step5b-audit-summary.json`. Git contains all locked-test and cross-dataset pairs, deterministic review samples, and the full aggregate counts.

See `docs/dataset-v2-acquisition-audit.md` for the reviewed Step 5B and Step 5B.1 findings.

## Audit the historical 39-class source

Step 5D validates the official non-augmented Mendeley archive, explicitly maps
all source folders to deployed indices, creates per-image SHA-256/dHash/pHash
metadata, and compares the source against final Dataset V2 candidates and the
locked PlantDoc TEST split:

```powershell
python scripts/audit_historical_39class_source.py
```

The verified archive contains 55,448 valid images across all 39 deployed
concepts. The logical clean candidate manifest contains 55,423 INCLUDE records,
23 excluded exact copies, and two unresolved interclass perceptual reviews.
No exact or verified perceptual overlap was found with Dataset V2 or PlantDoc
TEST. These findings recover full 39-class candidate coverage but do not prove
byte-level identity with the model's historical training directory. See
`docs/historical-39class-source-audit.md` for evidence, limitations, and the
complete coverage projection.

This command does not merge datasets, create splits, balance classes, augment
images, or train a model. Its large ORB signal report and pHash cache remain in
the ignored `local-audits/` directory.

## Build the logically cleaned manifests

Step 5C applies deterministic cleaning decisions without changing any source image:

```powershell
python scripts/build_dataset_v2_manifest.py
```

The command generates:

- `manifests/dataset-v2-master.csv`, containing all audited records and cleaning decisions;
- `manifests/dataset-v2-clean-candidates.csv`, containing only current `INCLUDE` records;
- `reports/exact-label-conflicts.csv`;
- `reports/perceptual-group-members.csv`;
- `reports/perceptual-groups-summary.csv`;
- `reports/perceptual-review-queue.csv`;
- `reports/dataset-v2-cleaning-summary.json`.

The ignored 288 MB pairwise dHash report is only needed to regenerate perceptual groups from scratch. Once compact group membership is generated, normal rebuilds and CI require no raw images, network access, or TensorFlow.

No train/validation split exists yet. Records sharing an `exact_duplicate_group_id` or `near_duplicate_group_id` must remain together when the future split is designed. See `docs/dataset-v2-cleaning-policy.md` for the complete policy and current counts.

## Refine perceptual screening

Step 5C.1 keeps dHash as candidate generation but verifies identity using a
64-bit DCT pHash, aspect-ratio geometry, ORB features, and RANSAC consistency:

```powershell
python scripts/refine_dataset_v2_perceptual_groups.py
python scripts/build_dataset_v2_manifest.py
```

The refinement creates compact `refined-near-duplicate-groups.csv`,
`refined-group-members.csv`, `perceptual-human-review.csv`, and
`perceptual-resolution-summary.json` reports. The master manifest then includes
`phash`, `review_resolution`, `refined_similarity_status`, `refined_group_id`,
and `refined_group_representative`.

The historical Step 5C dHash reports remain unchanged. Step 5D must use
`refined_group_id`, not the old transitive `near_duplicate_group_id`, as the
indivisible grouping constraint. See
`docs/dataset-v2-perceptual-resolution.md` for thresholds, before/after counts,
benchmark decisions, and remaining human-review cases.

## Final candidate pool

Step 5C.2 conservatively excludes only the 38 Step 5C.1 records whose visual
identity remained unresolved. Rebuild the final deterministic manifests with:

```powershell
python scripts/build_dataset_v2_manifest.py
```

The command writes `dataset-v2-final-candidate-summary.json`. The final semantic
pool contains 18,140 INCLUDE, zero REVIEW, and 1,164 EXCLUDE records. Dataset V2
currently represents 16 of the 39 deployed classes, so it is not by itself a
valid source for full 39-output retraining. No split or training has occurred.
See `docs/dataset-v2-final-candidate-pool.md`.

## Compose the final 39-class candidate pool

Step 5E combines only the two committed clean manifests: 55,423 controlled
historical candidates and 18,140 real-world Dataset V2 candidates. It also
finalizes the two historical review records as conservative metadata-only
exclusions and prepares indivisible grouping metadata for the future split:

```powershell
python scripts/build_dataset_v2_39class_composition.py
```

The command writes:

- `manifests/dataset-v2-39class-combined.csv` — 73,563 final candidates with
  complete provenance, authoritative target indices, and `split_group_id`;
- `reports/dataset-v2-39class-composition-summary.json` — class, domain,
  source, imbalance, group, collision, and benchmark metrics;
- `reports/historical-review-finalization.json` — the explicit Step 5E
  decisions for the two records formerly under review.

The combined pool covers 39/39 deployed classes. PlantDoc TEST remains locked,
and no known exact or verified perceptual cross-domain collision is admitted.
The command does not decode raw images, assign TRAIN/VALIDATION, materialize
directories, balance classes, augment data, or train a model. See
`docs/dataset-v2-39class-composition.md`.

## Group-aware TRAIN / VALIDATION / TEST split

Step 5F partitions the authoritative Step 5E composition by indivisible
`split_group_id` with seed `20260810` and strategy
`dataset-v2-group-aware-80-10-10-v1`:

```powershell
python scripts/build_dataset_v2_group_aware_split.py
```

The metadata-only command generates:

- `manifests/dataset-v2-39class-split.csv` — all 73,563 records with split,
  seed, strategy, and evaluation role;
- `manifests/dataset-v2-train.csv` — TRAIN records only;
- `manifests/dataset-v2-validation.csv` — the only internal holdout permitted
  to guide model-development decisions;
- `manifests/dataset-v2-test.csv` — the frozen final internal test;
- `reports/dataset-v2-39class-split-summary.json` — class, domain, source,
  group, leakage, and holdout-quality metrics;
- `reports/dataset-v2-39class-split-quality.csv` — one audit row per deployed
  class;
- `reports/dataset-v2-test-lock.json` — the immutable TEST identity and policy.

The achieved split is 58,857 TRAIN, 7,362 VALIDATION, and 7,344 TEST images,
with 39/39 class coverage in every partition and zero group, SHA-256, or
PlantDoc TEST leakage. Do not use the TEST manifest for early stopping,
threshold tuning, augmentation, sampling, fine-tuning, or model selection.
PlantDoc TEST remains a separate external benchmark. Balancing, augmentation,
and training are outside Step 5F. See `docs/dataset-v2-group-aware-split.md`.
