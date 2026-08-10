# Dataset V2 final candidate pool

## Scope

Step 5C.2 finalizes deterministic dataset membership. It does not relabel an
image, delete a source file, balance a class, create a train/validation split,
or train a model. The scientific Step 5C.1 reports remain unchanged and retain
the original similarity metrics, review identifiers, and `STILL_UNCERTAIN`
decisions.

## Conservative resolution policy

After exact-duplicate, benchmark-leakage, dHash, pHash, geometry, ORB, and
RANSAC analysis, 38 PLDD-UP records in 13 identity cases remained unresolved.
Only those records receive the final decision:

```text
cleaning_status = EXCLUDE
exclusion_reason = UNRESOLVED_PERCEPTUAL_IDENTITY
review_resolution = CONSERVATIVE_FINAL_EXCLUSION
```

This decision means that the observation is unsafe for deterministic splitting
and training. It does not mean that its source disease label is wrong. No
disease was inferred visually and no image was relabeled.

The 38 records are 0.196850% of the 19,304-record semantic pool. Losing this
small quantity is preferable to carrying unresolved identity conflicts into a
split: membership becomes deterministic, leakage risk is reduced, and no
subjective disease diagnosis is introduced. The 2,694 false-positive dHash
reviews restored in Step 5C.1 remain included.

## Final decisions

| Decision | Step 5C.1 | Step 5C.2 |
|---|---:|---:|
| INCLUDE | 18,140 | 18,140 |
| REVIEW | 38 | 0 |
| EXCLUDE | 1,126 | 1,164 |

The final clean-candidate manifest contains exactly 18,140 INCLUDE rows and no
REVIEW, EXCLUDE, locked-test, ambiguous-mapping, unsupported-class, exact
duplicate, confirmed label-conflict, or confirmed benchmark-leakage row.
PlantDoc TEST remains locked and unchanged.

## Counts by source

| Source | Raw semantic | INCLUDE | EXCLUDE | REVIEW |
|---|---:|---:|---:|---:|
| PlantDoc TRAIN | 1,085 | 1,034 | 51 | 0 |
| Banu/Deb Potato originals | 36 | 24 | 12 | 0 |
| Seasonal Corn originals | 2,664 | 2,047 | 617 | 0 |
| PLDD-UP | 15,519 | 15,035 | 484 | 0 |

The 38 final exclusions affect Potato Early blight (11), Potato Late blight
(13), and Potato healthy (14). Their final INCLUDE counts remain 4,751, 6,105,
and 4,388 respectively across all sources.

## Deployed-class coverage

Dataset V2 contains final candidates for 16 of the 39 deployed outputs:

- Apple Apple scab
- Corn Cercospora leaf spot
- Corn Common rust
- Corn healthy
- Grape Black rot
- Potato Early blight
- Potato Late blight
- Potato healthy
- Squash Powdery mildew
- Tomato Bacterial spot
- Tomato Early blight
- Tomato Late blight
- Tomato Leaf Mold
- Tomato Septoria leaf spot
- Tomato Spider mites
- Tomato mosaic virus

The following 23 deployed classes have no final Dataset V2 candidate:

- Apple Black rot
- Apple Cedar apple rust
- Apple healthy
- Background without leaves
- Bell pepper Bacterial spot
- Bell pepper healthy
- Blueberry healthy
- Cherry Powdery mildew
- Cherry healthy
- Corn Northern Leaf Blight
- Grape Esca
- Grape Leaf blight
- Grape healthy
- Orange Huanglongbing
- Peach Bacterial spot
- Peach healthy
- Raspberry healthy
- Soybean healthy
- Strawberry Leaf scorch
- Strawberry healthy
- Tomato Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato healthy

## Critical 39-class training gate

The cleaned Dataset V2 pool alone cannot retrain a full 39-output classifier:
23 output classes, including the background/no-leaf class, would have no
training examples. Before a full Model V2 training plan is approved, the
historical PlantVillage-style source used for the deployed taxonomy must be
recovered or reproducibly reconstructed and audited. This task does not solve
that data-coverage problem.

## Reproduction and next gate

Run:

```powershell
python scripts/build_dataset_v2_manifest.py
```

The command regenerates the master manifest, clean candidate manifest, and
`dataset-v2-final-candidate-summary.json` from committed compact audit
artifacts. No network, raw dataset, or TensorFlow model is required for this
logical rebuild.

Step 5D may design a split only after human approval of this final pool and a
decision on the missing 23-class source requirement. No split, balancing,
augmentation, fine-tuning, or model training has been performed.
