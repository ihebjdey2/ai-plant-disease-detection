# Dataset V2 group-aware 80/10/10 split

## Purpose

Step 5F partitions the final Step 5E candidate pool into three immutable roles:

- **TRAIN (80% target)** learns model weights.
- **VALIDATION (10% target)** is the only internal holdout allowed to guide model-development decisions, including early stopping, checkpoint selection, learning-rate policy, augmentation comparisons, sampling/class-weight comparisons, fine-tuning depth, calibration, and threshold work.
- **INTERNAL TEST (10% target)** is a frozen, final in-distribution evaluation set across all 39 deployed classes. It must not guide development.

The previous 85/15 proposal was superseded because 73,563 audited candidates are sufficient to reserve a full 39-class internal test while retaining substantial training and validation sets.

## Internal TEST versus PlantDoc TEST

These are two different evaluation layers:

- INTERNAL TEST is sampled from the composed Dataset V2 distribution and covers all 39 deployed classes.
- PlantDoc TEST remains outside Dataset V2 and measures external-domain generalization on its verified overlapping subset.

PlantDoc TEST did not influence the seed, allocation, ratios, or model-development holdouts. The generated manifests contain zero PlantDoc TEST records.

## Deterministic allocation

The builder is `scripts/build_dataset_v2_group_aware_split.py`. Its only candidate input is:

`training/datasets/manifests/dataset-v2-39class-combined.csv`

The split is defined by:

- seed: `20260810`;
- strategy: `dataset-v2-group-aware-80-10-10-v1`;
- atomic unit: `split_group_id`;
- target ratios: 80% TRAIN, 10% VALIDATION, 10% TEST.

Before allocation, each group is verified to contain exactly one target index, target class, and source domain. Complete groups are stratified by target class, source domain, and source-dataset membership. A seeded SHA-256 ordering provides stable tie-breaking without depending on input or filesystem iteration order.

The allocator minimizes image-count and group-count target error without ever splitting a group. It applies this conservative group-support policy to each class/domain/source stratum:

- at least 20 independent groups: at least two VALIDATION groups and two TEST groups;
- 10–19 groups: at least one VALIDATION group and one TEST group;
- fewer than 10 groups: TRAIN-only, to preserve scarce training utility rather than manufacture misleading holdouts.

This last rule explains why the two real-world Tomato Spider mites images remain in TRAIN. Class-level VALIDATION and TEST coverage still comes from the controlled historical domain.

## Achieved split

| Partition | Images | Percentage | Groups | Class coverage |
|---|---:|---:|---:|---:|
| TRAIN | 58,857 | 80.00897190% | 56,478 | 39/39 |
| VALIDATION | 7,362 | 10.00774846% | 7,098 | 39/39 |
| TEST | 7,344 | 9.98327964% | 7,087 | 39/39 |
| Total | 73,563 | 100% | 70,663 | 39/39 |

Group integrity takes precedence over exact percentages. The achieved deviations are only +0.00897, +0.00775, and −0.01672 percentage points, respectively.

## Domain and source preservation

| Domain | Total | TRAIN | VALIDATION | TEST |
|---|---:|---:|---:|---:|
| HISTORICAL_CONTROLLED | 55,423 | 44,340 | 5,546 | 5,537 |
| REAL_WORLD | 18,140 | 14,517 | 1,816 | 1,807 |

| Real-world source | Total | TRAIN | VALIDATION | TEST |
|---|---:|---:|---:|---:|
| PLDD-UP | 15,035 | 12,030 | 1,504 | 1,501 |
| Seasonal Corn | 2,047 | 1,638 | 205 | 204 |
| PlantDoc TRAIN-source | 1,034 | 829 | 105 | 100 |
| Banu/Deb Potato | 24 | 20 | 2 | 2 |

The committed JSON report is authoritative if documentation figures ever need to be regenerated. Counts above are generated from strategy v1 and seed 20260810.

## Real-world holdout interpretation

Holdout quality uses objective criteria:

- `ROBUST_REAL_WORLD_HOLDOUT`: at least 20 real-world images and two independent real-world groups in both VALIDATION and TEST;
- `LIMITED_REAL_WORLD_HOLDOUT`: some real-world holdout exists, but the robust threshold is not met;
- `NO_REAL_WORLD_HOLDOUT`: no real-world image is available in either holdout;
- `NOT_APPLICABLE_NO_REAL_WORLD_DATA`: the class has no real-world candidates.

Five classes have robust real-world holdouts: Corn Cercospora leaf spot, Corn healthy, Potato Early blight, Potato Late blight, and Potato healthy. Ten classes have limited real-world holdouts. Tomato Spider mites has no real-world holdout because its two independent real-world examples remain in TRAIN. The other 23 deployed classes have no real-world source data and rely on controlled historical holdouts.

## Leakage and reconciliation invariants

The builder fails if any of these invariants is violated:

- every `composition_record_id` appears exactly once;
- every `split_group_id` belongs to exactly one partition;
- every SHA-256 content identity belongs to exactly one partition;
- all target indices 0–38 occur in every partition;
- no PlantDoc TEST record enters any internal manifest.

For the generated split:

- group leakage: 0;
- SHA-256 leakage: 0;
- PlantDoc TEST contamination: 0;
- missing or duplicated composition records: 0.

The `evaluation_role` column marks TEST rows as `FINAL_INTERNAL_TEST`. The companion `dataset-v2-test-lock.json` records the seed, strategy, record count, source-composition hash, and exact TEST-manifest hash.

## Rebuild

From the project root:

```powershell
python scripts/build_dataset_v2_group_aware_split.py
```

No TensorFlow, Internet access, raw-image decoding, or materialized image directories are required. Rebuilding twice from the same bytes, strategy, and seed must produce byte-identical manifests and reports.

## Development policy

Future training loaders may consume TRAIN and VALIDATION during model development. They must not load TEST for training-policy or model-selection decisions. Only after a final candidate has been selected using VALIDATION may it be evaluated once on INTERNAL TEST. External PlantDoc TEST remains separately locked.

VALIDATION and TEST retain their natural distributions. They must never be augmented, over-sampled, or under-sampled. Step 5F also leaves TRAIN unbalanced and unaugmented; any future balancing or augmentation policy applies only to TRAIN and requires separate human approval.

## Limitations

- Group-aware greedy allocation approximates, rather than forces, exact image ratios.
- Twenty-three classes still lack real-world support.
- Several real-world class holdouts are too small to be called robust.
- The two real-world Tomato Spider mites examples cannot support meaningful real-world validation and test estimates.
- The split prevents known SHA and trusted duplicate-family leakage represented by Step 5E metadata; it cannot detect an unknown visual relationship absent from the audited grouping data.
- No result in this step measures model accuracy: no balancing, augmentation, fine-tuning, or training was performed.
