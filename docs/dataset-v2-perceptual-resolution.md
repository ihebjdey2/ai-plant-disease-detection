# Dataset V2 perceptual review resolution

## Purpose and scope

Step 5C.1 corrects the perceptual over-grouping discovered during Step 5C. It resolves image-identity questions only. It does not visually diagnose disease, modify raw images, create a train/validation split, balance classes, augment data, or train a model.

PlantDoc TEST remains a locked benchmark. It is read only for leakage detection and is never eligible for training or validation.

## Why dHash was useful but insufficient

dHash is a fast 64-bit description of neighboring brightness changes. It efficiently reduced the search space and remains the first-stage candidate generator at Hamming distance 4 or lower. It is not proof that two images show the same observation.

Step 5C converted every dHash relation into unrestricted connected components. This allowed transitive chaining:

```text
A resembles B
B resembles C
C resembles D
```

to place `A`, `B`, `C`, and `D` in one group even when `A` and `D` had no direct evidence of similarity. The original Step 5C reports remain versioned as the historical screening result.

## The 2,763-image PLDD-UP failure mode

Before refinement, `near_2791da3a50be06fd2718` contained:

| Target | Records |
|---|---:|
| Potato Early blight | 1,084 |
| Potato Late blight | 1,069 |
| Potato healthy | 610 |
| **Total** | **2,763** |

The component had only two dominant resolutions:

- 1,662 images at 8,160 × 3,672;
- 1,101 images at 1,600 × 719.

Their aspect ratios were almost identical, from 2.2222 to 2.2253. This standardized panoramic geometry and coarse brightness structure caused dHash collisions across unrelated observations. Chaining then expanded those collisions into one artificial component.

The evidence is clear: all 1,441,693 relations inside the component had dHash distance 0–4, but their pHash distance had a median of 30, a 95th percentile of 38, and a maximum of 56. Actual burst photographs, resizes, and recompressions also exist, so the cause is a combination of real repetition, coarse-hash collision, homogeneous capture geometry, and transitive chaining—not 2,763 genuine duplicates.

## Stronger verification pipeline

The refined pipeline is:

```text
SHA-256 exact policy
        ↓
dHash ≤ 4 candidate screening
        ↓
64-bit DCT pHash
        ↓
aspect-ratio geometry check
        ↓
ORB feature matching + RANSAC geometric consistency
        ↓
direct-to-representative grouping
```

No TensorFlow model or neural embedding is used.

### pHash

The implementation converts the image to grayscale, resizes it to 32 × 32, computes a two-dimensional DCT, and thresholds the upper-left 8 × 8 low-frequency coefficients around their median. The result is a deterministic 64-bit hexadecimal hash.

The observed distributions justified two analysis bands:

- same-target grouping candidates: pHash distance ≤ 4;
- different-target and benchmark candidates: pHash distance ≤ 16.

The high-risk range was examined through 16 bits because verified ORB matches still occurred at distances 14 and 16. Above 16, the candidate distribution had moved into the broad collision population and was not treated as verified identity evidence.

### Geometry

The scale-invariant geometry check uses:

```text
abs(log(aspect_ratio_a / aspect_ratio_b)) ≤ 0.03
```

This allows normal resizing but rejects substantial aspect-ratio changes. Width and height may differ because recompressed and resized copies are expected.

### ORB and RANSAC

ORB is computed deterministically on grayscale images with a maximum dimension of 512 pixels and up to 800 features. Lowe-ratio-filtered feature matches are then checked using RANSAC homography consistency.

A relation is strongly verified only when all conditions hold:

- at least 40 good ORB matches;
- ORB match ratio ≥ 0.15;
- RANSAC inlier ratio ≥ 0.60;
- pHash and geometry conditions above.

The observed ORB population was strongly bimodal: verified copies exceeded these limits clearly, while ordinary dHash collisions generally had match ratios near zero.

A narrow intermediate band remains human review:

- at least 15 good matches;
- match ratio ≥ 0.05;
- inlier ratio ≥ 0.50;
- below the verified threshold.

All other different-target dHash candidates are treated as false-positive screening, not duplicates.

## Candidate verification and refined grouping

Only 15,332 of the original 1,445,318 dHash relations required ORB analysis:

- 3,527 same-target relations;
- 11,801 different-target relations;
- 4 TRAIN-to-PlantDoc-TEST relations.

Strong verification found 3,686 direct relations:

| Relation type | Strong relations |
|---|---:|
| Same target | 3,519 |
| Different target | 163 |
| TRAIN ↔ PlantDoc TEST | 4 |

Final groups use deterministic anchor grouping. Records are processed by stable `record_id`; the lowest eligible identifier becomes the representative. Every member must have a direct verified relation to that representative. A chain through another member cannot merge groups.

This produced 2,669 refined groups. The largest contains eight records.

### Giant component after refinement

The original 2,763-record component became:

- 555 verified refined groups;
- 1,224 records in verified groups;
- 1,539 records without a verified near-duplicate group;
- largest refined group: 5 records.

Group sizes are 466 pairs, 65 groups of three, 23 groups of four, and one group of five. Mixed-target verified groups remain conflicts and are excluded without selecting a disease label.

## PlantDoc TEST decisions

All four non-byte-identical TRAIN↔TEST pairs are verified as likely the same visual observation:

| TRAIN record | TEST record | pHash | ORB good | Match ratio | Inliers | Decision |
|---|---|---:|---:|---:|---:|---|
| Tomato Early blight `earlyblightpotato.jpg` | Potato Early blight `earlyblightpotato.jpg` | 0 | 800 | 1.000 | 1.000 | Exclude TRAIN |
| Tomato Late blight `late_blight1.jpg` | Tomato Late blight `18c.jpg` | 0 | 411 | 0.514 | 0.978 | Exclude TRAIN |
| Tomato Late blight foliar-lesions image | Potato Late blight image | 0 | 560 | 0.700 | 0.907 | Exclude TRAIN |
| Tomato bacterial-canker source | Tomato bacterial-spot resized image | 2 | 347 | 0.434 | 0.942 | Exclude TRAIN |

Four Step 5C review records therefore become `EXCLUDE / VERIFIED_PERCEPTUAL_BENCHMARK_LEAKAGE`. One additional training record belongs to those verified benchmark groups but was already excluded by a higher-priority exact rule. PlantDoc TEST rows remain `LOCKED_BENCHMARK` and untouched.

## Review resolution

Starting semantic decisions:

| Decision | Step 5C |
|---|---:|
| INCLUDE | 15,446 |
| REVIEW | 2,971 |
| EXCLUDE | 887 |

Transitions:

| Transition | Records |
|---|---:|
| REVIEW → INCLUDE | 2,694 |
| REVIEW → EXCLUDE | 239 |
| REVIEW → REVIEW | 38 |

Final semantic decisions:

| Decision | Step 5C.1 |
|---|---:|
| INCLUDE | 18,140 |
| REVIEW | 38 |
| EXCLUDE | 1,126 |

The 2,694 restored records are false-positive dHash conflicts. New verified exclusions consist of 235 training records with perceptual label conflicts and four training records with benchmark leakage. The 38 remaining records are represented by 13 compact PLDD-UP identity-review cases.

## Counts by source

| Source | Raw semantic | INCLUDE | REVIEW | EXCLUDE |
|---|---:|---:|---:|---:|
| PlantDoc TRAIN | 1,085 | 1,034 | 0 | 51 |
| Banu/Deb Potato originals | 36 | 24 | 0 | 12 |
| Seasonal Corn originals | 2,664 | 2,047 | 0 | 617 |
| PLDD-UP | 15,519 | 15,035 | 38 | 446 |

PLDD-UP changed from 12,342 INCLUDE / 2,918 REVIEW / 259 EXCLUDE to 15,035 INCLUDE / 38 REVIEW / 446 EXCLUDE.

PLDD-UP final class decisions are:

| Class | Raw | INCLUDE | REVIEW | EXCLUDE | Refined groups |
|---|---:|---:|---:|---:|---:|
| Potato Early blight | 4,803 | 4,656 | 11 | 136 | 325 |
| Potato Late blight | 6,116 | 5,995 | 13 | 108 | 401 |
| Potato healthy | 4,600 | 4,384 | 14 | 202 | 1,484 |

## Final counts by deployed target

| Target | Raw | INCLUDE | REVIEW | EXCLUDE | Refined groups | Sources |
|---|---:|---:|---:|---:|---:|---:|
| Apple Apple scab | 83 | 83 | 0 | 0 | 1 | 1 |
| Corn Cercospora leaf spot | 1,561 | 950 | 0 | 611 | 62 | 2 |
| Corn Common rust | 129 | 129 | 0 | 0 | 2 | 1 |
| Corn healthy | 1,038 | 1,030 | 0 | 8 | 432 | 1 |
| Grape Black rot | 56 | 56 | 0 | 0 | 1 | 1 |
| Potato Early blight | 4,912 | 4,751 | 11 | 150 | 336 | 2 |
| Potato Late blight | 6,233 | 6,105 | 13 | 115 | 409 | 3 |
| Potato healthy | 4,616 | 4,388 | 14 | 214 | 1,484 | 2 |
| Squash Powdery mildew | 124 | 124 | 0 | 0 | 2 | 1 |
| Tomato Bacterial spot | 101 | 98 | 0 | 3 | 4 | 1 |
| Tomato Early blight | 79 | 69 | 0 | 10 | 12 | 1 |
| Tomato Late blight | 101 | 93 | 0 | 8 | 11 | 1 |
| Tomato Leaf Mold | 85 | 85 | 0 | 0 | 0 | 1 |
| Tomato Septoria leaf spot | 140 | 133 | 0 | 7 | 9 | 1 |
| Tomato Spider mites | 2 | 2 | 0 | 0 | 0 | 1 |
| Tomato mosaic virus | 44 | 44 | 0 | 0 | 0 | 1 |

No class balancing was performed.

## Reproducible commands

With the audited raw sources and ignored exhaustive dHash report available:

```powershell
python scripts/refine_dataset_v2_perceptual_groups.py
python scripts/build_dataset_v2_manifest.py
```

The first command regenerates pHash/ORB signals and compact refined reports without network access or TensorFlow. The second rebuilds master and clean manifests from committed compact artifacts; normal CI does not require raw images or the exhaustive pair report.

## Remaining limitations

Thirteen representative PLDD-UP cases covering 38 records remain unresolved. They must receive a human image-identity decision before Step 5D. No disease label should be inferred visually. pHash and ORB remain engineered similarity signals rather than a proof of semantic identity, and thresholds must not be reused on unrelated datasets without a new distribution study.

No split has been created. Step 5D must treat each non-empty `refined_group_id` as indivisible.
