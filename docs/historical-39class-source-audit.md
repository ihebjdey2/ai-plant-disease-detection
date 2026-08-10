# Historical 39-class source recovery and audit (Step 5D)

## Scope and decision

Dataset V2 contains 18,140 final real-world candidates but represents only 16
of the 39 deployed outputs. It cannot independently support a new 39-output
classifier. Step 5D therefore recovered and audited the most likely controlled
historical source while deliberately stopping before dataset composition,
splitting, balancing, augmentation, or training.

The provenance classification is **`STRONG_MATCH_BUT_NOT_PROVEN`**. The
hash-verified official non-augmented archive contains exactly the same 55,448
images and 39 classes reported by the notebook generators, and it matches the
same PlantVillage-style directory concept. The notebook does not, however,
persist an archive hash, source file identifier, full `class_indices` output,
or per-image manifest. Those missing identifiers prevent a scientifically
defensible `CONFIRMED_EXACT_SOURCE` claim.

## Official source and archive integrity

| Field | Verified value |
| --- | --- |
| Dataset | Data for: Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network |
| Authors | Arun Pandian J; Geetharamani Gopal |
| DOI / version | `10.17632/tywbtsjrjv.1`, version 1 |
| Publication date | 2019-04-18 |
| License | CC0 1.0 |
| Retrieval date | 2026-08-10 |
| Official file | `Plant_leaf_diseases_dataset_without_augmentation.zip` |
| Official file ID | `d5652a28-c1d8-4b76-97f3-72fb80f94efc` |
| Compressed size | 868,032,562 bytes |
| SHA-256 | `ac3432453984d02a86197987e775a5429d0d59e7cc7c35bcf5a8f50349b90ff0` |
| ZIP entries / files | 55,488 / 55,448 |
| Unsafe paths / CRC errors | 0 / 0 |

Only the official [Mendeley Data version 1](https://data.mendeley.com/datasets/tywbtsjrjv/1)
non-augmented archive was downloaded. The separately published augmented
archive was not downloaded or inspected, because pre-generated transformations
could contaminate a future validation split.

The drive had 188.67 GiB free before acquisition and 186.68 GiB after the
archive, immutable extraction, and local audit caches were present. The initial
3 GiB conservative temporary-space allowance was therefore safe.

## Verified non-augmented contents

The publication-level description reports 61,486 images after applying six
augmentation methods. That is not the count of the non-augmented archive. The
downloaded non-augmented archive independently verifies as follows:

- image files: 55,448;
- valid images: 55,448;
- corrupt images: 0;
- unsupported files: 0;
- source class folders: 39;
- formats: 55,447 JPEG and 1 PNG;
- modes: 55,447 RGB and 1 RGBA;
- dimensions: width 256–256 pixels, height 192–256 pixels.

The 6,038-image difference between the publication-level count and the
non-augmented archive is **likely** associated with the publication's augmented
variant, but this audit does not claim that as confirmed because the augmented
archive was intentionally not acquired. The equality between the notebook
generator total and the verified non-augmented count is confirmed.

## Complete taxonomy, explicit mapping, and coverage projection

External folder order was never used to derive output indices. Every row below
comes from the reviewed mapping and is validated against `app/taxonomy.py`.
`Historical clean` excludes 23 exact duplicate copies and the two unresolved
interclass perceptual candidates. `Projected` is only arithmetic; no combined
manifest was created.

| Index | Official source folder | Raw | Deployed class | Historical clean | Real-world V2 | Projected |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 0 | `Apple___Apple_scab` | 630 | Apple Apple scab | 630 | 83 | 713 |
| 1 | `Apple___Black_rot` | 621 | Apple Black rot | 621 | 0 | 621 |
| 2 | `Apple___Cedar_apple_rust` | 275 | Apple Cedar apple rust | 275 | 0 | 275 |
| 3 | `Apple___healthy` | 1,645 | Apple healthy | 1,638 | 0 | 1,638 |
| 4 | `Background_without_leaves` | 1,143 | Background without leaves | 1,141 | 0 | 1,141 |
| 5 | `Blueberry___healthy` | 1,502 | Blueberry healthy | 1,502 | 0 | 1,502 |
| 6 | `Cherry___Powdery_mildew` | 1,052 | Cherry Powdery mildew | 1,052 | 0 | 1,052 |
| 7 | `Cherry___healthy` | 854 | Cherry healthy | 854 | 0 | 854 |
| 8 | `Corn___Cercospora_leaf_spot Gray_leaf_spot` | 513 | Corn Cercospora leaf spot | 513 | 950 | 1,463 |
| 9 | `Corn___Common_rust` | 1,192 | Corn Common rust | 1,192 | 129 | 1,321 |
| 10 | `Corn___Northern_Leaf_Blight` | 985 | Corn Northern Leaf Blight | 985 | 0 | 985 |
| 11 | `Corn___healthy` | 1,162 | Corn healthy | 1,162 | 1,030 | 2,192 |
| 12 | `Grape___Black_rot` | 1,180 | Grape Black rot | 1,180 | 56 | 1,236 |
| 13 | `Grape___Esca_(Black_Measles)` | 1,383 | Grape Esca | 1,383 | 0 | 1,383 |
| 14 | `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | 1,076 | Grape Leaf blight | 1,076 | 0 | 1,076 |
| 15 | `Grape___healthy` | 423 | Grape healthy | 423 | 0 | 423 |
| 16 | `Orange___Haunglongbing_(Citrus_greening)` | 5,507 | Orange Huanglongbing | 5,507 | 0 | 5,507 |
| 17 | `Peach___Bacterial_spot` | 2,297 | Peach Bacterial spot | 2,297 | 0 | 2,297 |
| 18 | `Peach___healthy` | 360 | Peach healthy | 360 | 0 | 360 |
| 19 | `Pepper,_bell___Bacterial_spot` | 997 | Bell pepper Bacterial spot | 997 | 0 | 997 |
| 20 | `Pepper,_bell___healthy` | 1,478 | Bell pepper healthy | 1,478 | 0 | 1,478 |
| 21 | `Potato___Early_blight` | 1,000 | Potato Early blight | 1,000 | 4,751 | 5,751 |
| 22 | `Potato___Late_blight` | 1,000 | Potato Late blight | 1,000 | 6,105 | 7,105 |
| 23 | `Potato___healthy` | 152 | Potato healthy | 152 | 4,388 | 4,540 |
| 24 | `Raspberry___healthy` | 371 | Raspberry healthy | 371 | 0 | 371 |
| 25 | `Soybean___healthy` | 5,090 | Soybean healthy | 5,090 | 0 | 5,090 |
| 26 | `Squash___Powdery_mildew` | 1,835 | Squash Powdery mildew | 1,835 | 124 | 1,959 |
| 27 | `Strawberry___Leaf_scorch` | 1,109 | Strawberry Leaf scorch | 1,109 | 0 | 1,109 |
| 28 | `Strawberry___healthy` | 456 | Strawberry healthy | 456 | 0 | 456 |
| 29 | `Tomato___Bacterial_spot` | 2,127 | Tomato Bacterial spot | 2,127 | 98 | 2,225 |
| 30 | `Tomato___Early_blight` | 1,000 | Tomato Early blight | 1,000 | 69 | 1,069 |
| 31 | `Tomato___Late_blight` | 1,909 | Tomato Late blight | 1,900 | 93 | 1,993 |
| 32 | `Tomato___Leaf_Mold` | 952 | Tomato Leaf Mold | 952 | 85 | 1,037 |
| 33 | `Tomato___Septoria_leaf_spot` | 1,771 | Tomato Septoria leaf spot | 1,771 | 133 | 1,904 |
| 34 | `Tomato___Spider_mites Two-spotted_spider_mite` | 1,676 | Tomato Spider mites | 1,676 | 2 | 1,678 |
| 35 | `Tomato___Target_Spot` | 1,404 | Tomato Target Spot | 1,404 | 0 | 1,404 |
| 36 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 5,357 | Tomato Yellow Leaf Curl Virus | 5,357 | 0 | 5,357 |
| 37 | `Tomato___Tomato_mosaic_virus` | 373 | Tomato mosaic virus | 373 | 44 | 417 |
| 38 | `Tomato___healthy` | 1,591 | Tomato healthy | 1,584 | 0 | 1,584 |

All 39 source labels are `MATCHED`, every deployed index appears exactly once,
and there are no ambiguous, unsupported, duplicated, or missing target
mappings. The clean historical candidate coverage is **39/39**.

## Notebook evidence and provenance comparison

| Finding | Classification | Evidence |
| --- | --- | --- |
| 39 output classes | CONFIRMED | Notebook generator output and final Dense(39) layer. |
| 55,448 source images | CONFIRMED | 44,371 training + 11,077 validation generator images. |
| RGB, 224×224, divide by 255 | CONFIRMED | Generator target size and `rescale=1./255`; matches deployed inference. |
| Validation strategy | CONFIRMED | `validation_split=0.2` with generator subsets. |
| Training augmentation | CONFIRMED | Rotation, shifts, shear, zoom, horizontal flip, nearest fill. |
| Validation augmentation | CONFIRMED | The same augmented generator supplies the validation subset. |
| Independent labeled TEST split | CONFIRMED absent | No independent test generator or labeled test evaluation is defined. |
| Historical directory | CONFIRMED | Explicit notebook path ends in `PlantVillage`; that path no longer exists locally. |
| Background present in exact notebook source | LIKELY | Official archive, total, and 39-concept taxonomy match, but notebook class indices are not persisted. |
| Historical output ordering from notebook | UNKNOWN | `class_indices` code exists but its output was not saved. |
| Exact Mendeley archive identity | STRONG MATCH, NOT PROVEN | Counts, concepts, directory purpose, and preprocessing agree; immutable source identity is missing. |

The notebook later labels a section as fine-tuning and freezes the first 100
MobileNetV2 layers. Since the base model was already frozen and the notebook
does not explicitly re-enable the later layers, the intended fine-tuning does
not independently strengthen source provenance.

## Background class audit

`Background_without_leaves` contains 1,143 valid JPEG images, all 256×192, in
the same hash-verified official archive. One exact group contains three images,
so two exact copies are excluded from the clean candidate manifest. Three
verified direct perceptual groups contain six members. The final clean
background candidate count is 1,141.

The label is an exact semantic match to deployed index 4. It is evidence for a
historical non-leaf background class, not evidence that this class is a
universal out-of-distribution detector. No application `no_leaf` behavior was
changed.

## Duplicate and leakage audit

The audit uses the Step 5C.1 sequence: SHA-256, lossless band-indexed dHash
screening at Hamming distance ≤4, 64-bit DCT pHash, aspect-ratio geometry, and
ORB/RANSAC verification. It does not build unrestricted transitive dHash
components.

### Internal historical source

- exact duplicate groups: 22;
- exact copies beyond the canonical member: 23;
- exact cross-class label conflicts: 0;
- dHash non-exact candidate pairs: 268;
- pairs reaching ORB/RANSAC after pHash and geometry: 59;
- verified same-target near-duplicate pairs: 22;
- direct representative groups / grouped images: 22 / 44;
- possible signals: 18;
- unresolved possible cross-class pair: 1 (`Tomato Late blight` ↔ `Tomato healthy`), placing both records in review.

Raw files were not deleted. Exact canonicalization and the two review decisions
exist only in metadata.

### Historical source ↔ final Dataset V2 INCLUDE

- exact SHA-256 overlapping historical images / pairs: 0 / 0;
- dHash candidates: 23;
- candidates passing pHash and geometry to ORB/RANSAC: 0;
- verified perceptual overlapping historical images / pairs: 0 / 0.

The result does not prove that the complete historical training material is
unrelated to every public source; it reports only the locally audited official
archive against the final 18,140 Dataset V2 INCLUDE records.

### Historical source ↔ locked PlantDoc TEST

- exact SHA-256 leakage: 0 images / 0 pairs;
- dHash candidates: 0;
- verified perceptual leakage: 0 images / 0 pairs;
- possible perceptual leakage: 0 images.

PlantDoc TEST remained read-only and was read from its locked Git revision. No
claim is made about leakage against an unavailable historical training archive
other than the one acquired here.

## Candidate policy and domain balance

| Status | Images |
| --- | ---: |
| `INCLUDE_CANDIDATE` | 55,423 |
| `EXCLUDE_EXACT_DUPLICATE` | 23 |
| `REVIEW_PERCEPTUAL_CONFLICT` | 2 |
| `EXCLUDE_LABEL_CONFLICT` | 0 |
| `EXCLUDE_BENCHMARK_LEAKAGE` | 0 |
| `INVALID_IMAGE` | 0 |

The projection combines counts arithmetically but does not merge records. Its
largest class is Potato Late blight (7,105), its smallest is Apple Cedar apple
rust (275), and its median class size is 1,383. Twenty-three classes are still
historical-only; 16 have real-world support; and four have multiple real-world
sources: Corn Cercospora leaf spot, Potato Early blight, Potato Late blight,
and Potato healthy. No balancing, sampling, augmentation, or class weighting
was attempted.

## Reproduction and determinism

With the official archive and immutable extraction present in ignored local
paths:

```powershell
python scripts/audit_historical_39class_source.py
```

Two consecutive final rebuilds produced byte-identical outputs. Representative
SHA-256 values are:

- source manifest: `b1d6420a5430aff43ee364e09774ad43fbeea300ecc50ae221d41db67c0d98ca`;
- clean manifest: `cec328797c275dc2049cd7a57facc8c63270d2099c919577c655ff1063f37c72`;
- audit summary: `8a4078119286e29497f0d2d3619b1b74bbe3fc35e3c0ba442f1f27e75169ebfc`.

The exhaustive ORB signal file and external pHash cache remain in ignored
`training/datasets/local-audits/`. Normal tests use synthetic data and require
neither the Mendeley download, TensorFlow, nor network access.

## Limitations and stop condition

- Byte-level identity between this archive and the original model's actual
  training directory remains unproven.
- The original notebook's class-index dictionary is unavailable, so deployed
  order continues to come only from the authoritative application taxonomy.
- Exact and perceptual audits reduce known overlap risk but cannot prove zero
  leakage against unavailable historical data or all internet mirrors.
- The notebook's augmented validation strategy is documented as historical
  behavior, not accepted as a future split design.
- No final historical + real-world manifest, train/validation split, balancing,
  runtime augmentation, fine-tuning, or training exists in this step.

Human approval is required before Step 5E defines the two-record perceptual
review outcome, cross-source canonicalization policy, group-aware validation
design, and final dataset composition.
