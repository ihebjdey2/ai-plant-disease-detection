# Final 39-class Dataset V2 composition (Step 5E)

## What this step does

The real-world Dataset V2 pool contains 18,140 audited candidates but covers
only 16 of the 39 deployed classes. The recovered historical Mendeley source
contains 55,423 clean controlled candidates and covers all 39 classes. Step 5E
combines these two **metadata manifests** into one deterministic candidate pool
without reading or copying raw images.

This is composition, not training. No train/validation assignment, balancing,
undersampling, oversampling, augmentation, class weighting, or model operation
occurs here.

## Authoritative inputs

- `historical-mendeley-39-clean-candidates.csv`: clean controlled historical
  candidates from the official non-augmented Mendeley archive;
- `dataset-v2-clean-candidates.csv`: final clean real-world candidates from
  PlantDoc TRAIN, Seasonal Corn, PLDD-UP, and the Banu/Deb Potato originals;
- the full historical manifest and both Step 5C.2/5D summaries, used to verify
  exclusions, benchmark safety, and input counts;
- `app/taxonomy.py`, which alone defines deployed indices 0–38.

Raw folders are not composition inputs. External folder order and CSV order
never determine target indices.

## Historical review finalization

Step 5D left two possible perceptual matches under review: one labeled Tomato
Late blight and one labeled Tomato healthy. Neither record was present in the
historical clean manifest. Step 5E finalizes both conservatively as:

```text
cleaning_status = EXCLUDE
exclusion_reason = UNRESOLVED_HISTORICAL_PERCEPTUAL_IDENTITY
review_resolution = CONSERVATIVE_FINAL_EXCLUSION
```

No disease label was selected, no relabeling occurred, and no raw file was
deleted. The historical clean count therefore remains 55,423 and historical
REVIEW becomes zero for final composition. The Step 5D manifest continues to
preserve the earlier review evidence; the Step 5E final decision is recorded in
`historical-review-finalization.json` and embedded in the composition summary.

## Final pool

| Domain/source | Candidates | Share |
| --- | ---: | ---: |
| Historical controlled — Mendeley 39-class | 55,423 | 75.340864% |
| Real-world — PLDD-UP | 15,035 | 20.438264% of total |
| Real-world — Seasonal Corn | 2,047 | 2.782649% of total |
| Real-world — PlantDoc TRAIN | 1,034 | 1.405326% of total |
| Real-world — Banu/Deb Potato originals | 24 | 0.032625% of total |
| **All real-world sources** | **18,140** | **24.659136%** |
| **Combined** | **73,563** | **100%** |

The result covers all 39 deployed indices and classes.

## Counts and domain composition by class

The domain percentages below are within each class. `Sources` is the number of
distinct real-world datasets supporting that class.

| Index | Deployed class | Historical | Real-world | Combined | Sources | Historical % | Real % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Apple Apple scab | 630 | 83 | 713 | 1 | 88.36 | 11.64 |
| 1 | Apple Black rot | 621 | 0 | 621 | 0 | 100.00 | 0.00 |
| 2 | Apple Cedar apple rust | 275 | 0 | 275 | 0 | 100.00 | 0.00 |
| 3 | Apple healthy | 1,638 | 0 | 1,638 | 0 | 100.00 | 0.00 |
| 4 | Background without leaves | 1,141 | 0 | 1,141 | 0 | 100.00 | 0.00 |
| 5 | Blueberry healthy | 1,502 | 0 | 1,502 | 0 | 100.00 | 0.00 |
| 6 | Cherry Powdery mildew | 1,052 | 0 | 1,052 | 0 | 100.00 | 0.00 |
| 7 | Cherry healthy | 854 | 0 | 854 | 0 | 100.00 | 0.00 |
| 8 | Corn Cercospora leaf spot | 513 | 950 | 1,463 | 2 | 35.06 | 64.94 |
| 9 | Corn Common rust | 1,192 | 129 | 1,321 | 1 | 90.23 | 9.77 |
| 10 | Corn Northern Leaf Blight | 985 | 0 | 985 | 0 | 100.00 | 0.00 |
| 11 | Corn healthy | 1,162 | 1,030 | 2,192 | 1 | 53.01 | 46.99 |
| 12 | Grape Black rot | 1,180 | 56 | 1,236 | 1 | 95.47 | 4.53 |
| 13 | Grape Esca | 1,383 | 0 | 1,383 | 0 | 100.00 | 0.00 |
| 14 | Grape Leaf blight | 1,076 | 0 | 1,076 | 0 | 100.00 | 0.00 |
| 15 | Grape healthy | 423 | 0 | 423 | 0 | 100.00 | 0.00 |
| 16 | Orange Huanglongbing | 5,507 | 0 | 5,507 | 0 | 100.00 | 0.00 |
| 17 | Peach Bacterial spot | 2,297 | 0 | 2,297 | 0 | 100.00 | 0.00 |
| 18 | Peach healthy | 360 | 0 | 360 | 0 | 100.00 | 0.00 |
| 19 | Bell pepper Bacterial spot | 997 | 0 | 997 | 0 | 100.00 | 0.00 |
| 20 | Bell pepper healthy | 1,478 | 0 | 1,478 | 0 | 100.00 | 0.00 |
| 21 | Potato Early blight | 1,000 | 4,751 | 5,751 | 2 | 17.39 | 82.61 |
| 22 | Potato Late blight | 1,000 | 6,105 | 7,105 | 3 | 14.07 | 85.93 |
| 23 | Potato healthy | 152 | 4,388 | 4,540 | 2 | 3.35 | 96.65 |
| 24 | Raspberry healthy | 371 | 0 | 371 | 0 | 100.00 | 0.00 |
| 25 | Soybean healthy | 5,090 | 0 | 5,090 | 0 | 100.00 | 0.00 |
| 26 | Squash Powdery mildew | 1,835 | 124 | 1,959 | 1 | 93.67 | 6.33 |
| 27 | Strawberry Leaf scorch | 1,109 | 0 | 1,109 | 0 | 100.00 | 0.00 |
| 28 | Strawberry healthy | 456 | 0 | 456 | 0 | 100.00 | 0.00 |
| 29 | Tomato Bacterial spot | 2,127 | 98 | 2,225 | 1 | 95.60 | 4.40 |
| 30 | Tomato Early blight | 1,000 | 69 | 1,069 | 1 | 93.55 | 6.45 |
| 31 | Tomato Late blight | 1,900 | 93 | 1,993 | 1 | 95.33 | 4.67 |
| 32 | Tomato Leaf Mold | 952 | 85 | 1,037 | 1 | 91.80 | 8.20 |
| 33 | Tomato Septoria leaf spot | 1,771 | 133 | 1,904 | 1 | 93.01 | 6.99 |
| 34 | Tomato Spider mites | 1,676 | 2 | 1,678 | 1 | 99.88 | 0.12 |
| 35 | Tomato Target Spot | 1,404 | 0 | 1,404 | 0 | 100.00 | 0.00 |
| 36 | Tomato Yellow Leaf Curl Virus | 5,357 | 0 | 5,357 | 0 | 100.00 | 0.00 |
| 37 | Tomato mosaic virus | 373 | 44 | 417 | 1 | 89.45 | 10.55 |
| 38 | Tomato healthy | 1,584 | 0 | 1,584 | 0 | 100.00 | 0.00 |

Twenty-three classes remain historical-only. Sixteen have real-world support.
Four have more than one real-world source: Corn Cercospora leaf spot, Potato
Early blight, Potato Late blight, and Potato healthy. This composition gap must
remain visible when future model generalization is interpreted.

## Duplicate and benchmark invariants

The builder revalidates zero SHA-256 collisions between historical and
real-world inputs. It also verifies that the Step 5D verified perceptual overlap
count remains zero. If either value becomes non-zero in a future rebuild, the
composition fails instead of inventing a canonicalization decision.

PlantDoc TEST remains outside both authoritative input manifests. Every
real-world row must have role `training_candidate`; benchmark flags, review
flags, non-matched mappings, non-INCLUDE decisions, and paths rooted at `test/`
are rejected. Final benchmark contamination is zero.

## Split-group metadata, not a split

The combined manifest contains `split_group_id`, but contains no split column
and no TRAIN or VALIDATION decision.

- real-world records reuse the refined Step 5C.1 group when available;
- historical records reuse the Step 5D perceptual group when available;
- exact groups are retained when no refined group supersedes them;
- every remaining image receives a stable singleton group;
- group identifiers are domain-scoped because no verified cross-domain image
  family exists.

The result contains 70,663 groups: 68,082 singletons and 2,581 multi-record
groups. The largest group has eight real-world Potato healthy records. No group
crosses a source domain or target class.

In Step 5F, all records sharing a group must move together. This prevents an
observation and its near-duplicate siblings from being divided between training
and validation.

## Measured imbalance

- minimum class size: 275 (Apple Cedar apple rust);
- maximum class size: 7,105 (Potato Late blight);
- median class size: 1,383;
- mean class size: 1,886.230769;
- maximum/minimum ratio: 25.836364.

The five smallest classes are Apple Cedar apple rust (275), Peach healthy
(360), Raspberry healthy (371), Tomato mosaic virus (417), and Grape healthy
(423). The five largest are Potato Late blight (7,105), Potato Early blight
(5,751), Orange Huanglongbing (5,507), Tomato Yellow Leaf Curl Virus (5,357),
and Soybean healthy (5,090).

No Potato image was removed because its class is large, and no rare-class image
was copied or augmented. Balancing is a later training-design decision.

## Rebuild

Composition needs only committed metadata and standard-library Python:

```powershell
python scripts/build_dataset_v2_39class_composition.py
```

It produces:

- `manifests/dataset-v2-39class-combined.csv`;
- `reports/dataset-v2-39class-composition-summary.json`;
- `reports/historical-review-finalization.json`.

Output records are sorted by deterministic composition ID. Repeated builds from
identical inputs must produce byte-identical manifest and summary hashes.

## Why Train/Validation is delayed

A naïve random image split would ignore observation families and could produce
optimistic validation metrics. Step 5F must jointly consider target class,
historical versus real-world domain, source dataset, class support, benchmark
integrity, and the indivisible `split_group_id`. Step 5E intentionally stops
before that design.
