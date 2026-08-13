# Model V2 final evaluation and provenance

## Scope

This document is the authoritative scientific record for the finalized
AgriDiagnose Model V2 candidate. It records the candidate-selection decision,
the one-time INTERNAL TEST estimate, and the external PlantDoc TEST benchmark
from frozen Kaggle artifacts. No training, inference, evaluation, threshold
change, or model update was performed while producing this record.

Model V2 is **frozen**. The selected candidate has not yet replaced the
application's deployed `plant_disease_model.h5` artifact.

## Candidate selection protocol

Experiments A and B used the same architecture, taxonomy, data splits,
preprocessing, optimizer policy, two-phase schedule, and selection rule. The
controlled Experiment B change was TRAIN-only augmentation. Overall VALIDATION
Macro-F1 was the primary candidate-selection metric. INTERNAL TEST and PlantDoc
TEST remained closed until the selection decision was complete.

| VALIDATION measure | Experiment A | Experiment B |
| --- | ---: | ---: |
| Overall Macro-F1 | 0.960108 | 0.958862 |
| Overall accuracy | 0.954768 | 0.953409 |
| Loss | 0.153806 | 0.154088 |
| Real-world-slice Macro-F1 | 0.645663 | 0.614674 |
| Real-world-slice accuracy | 0.886013 | 0.886013 |

For the real-world VALIDATION slice, 10,000 fixed-seed paired,
true-class-stratified bootstrap resamples placed the 95% confidence interval for
the Macro-F1 difference B minus A at `[-0.072920, 0.016322]`. Experiment B did
not improve the primary overall metric and did not demonstrate a reliable
real-world-slice gain. Experiment A was therefore **RETAINED / SELECTED**, and
Experiment B was **REJECTED**.

This was a VALIDATION-only decision. Neither TEST result was available or used
for candidate selection, and Experiment B was never evaluated on either TEST.

## Frozen model identity

- Selected experiment: `agri-diagnose-v2-exp-a`
- Rejected experiment: `agri-diagnose-v2-exp-b`
- Selected model SHA-256:
  `bba4d044bcafbbee6dcd9f604e9c3f10c42f2531f17f21769b991540e36b8ca0`
- Status: `FROZEN`

The checksum identifies the exact selected candidate. Test observations do not
authorize any change to this artifact.

## Final INTERNAL TEST

The selected Experiment A model was frozen before the Dataset V2 INTERNAL TEST
was opened. The evaluation used the locked `TEST` manifest containing 7,344
images across all 39 deployed classes. The split contained no PlantDoc TEST
records.

- Manifest SHA-256:
  `f0df59c42268163d485feea0e54dd7780aa56fe08a7984ae7869a09c604a9151`
- Accuracy: **95.44%** (`0.9543845315904139`)
- Macro-F1: **95.92%** (`0.9592043738824634`)
- Weighted-F1: **95.43%** (`0.954331728063991`)
- Keras loss: `0.15016824007034302`
- Predictions below the 60% confidence threshold: **2.14%**
  (`2.1377995642701526%`, 157 images)

The result is close to VALIDATION and supports stable generalization within the
Dataset V2 distribution. It does **not** prove equivalent performance on field
images, unseen capture devices, or different agricultural environments.

## External PlantDoc TEST — conservative 12-class semantic overlap subset

The external benchmark used the official TEST split from
`pratikkayal/PlantDoc-Dataset` at source revision
`5467f6012d78d1c446145d5f582da6096f852ae8`.

The official split contains 236 images in 27 original labels, with zero
corrupted images in the frozen audit. The existing conservative mapping accepted
only 12 exact semantic label matches and excluded 15 ambiguous labels. This
selected 99 images representing 12 of the 39 deployed classes. No exact
duplicate group was found and no duplicate image was removed.

- Accuracy: **41.41%** (`0.41414141414141414`)
- Macro-F1: **45.79%** (`0.45789013014076957`)
- Weighted-F1: **43.35%** (`0.4335358321873086`)
- Predictions below the 60% confidence threshold: **34.34%**
  (`34.34343434343434%`, 34 images)

This is a conservative **99-image / 12-class partial external benchmark**, not a
full 39-class PlantDoc accuracy result. It does not evaluate the deployed
background/no-leaf rejection class. Experiment B TEST evaluation remains
`false`.

## What each partition means

| Partition | Scientific role | Used for selection? |
| --- | --- | --- |
| VALIDATION | Checkpoint and Experiment A/B selection | Yes |
| Dataset V2 INTERNAL TEST | Final in-distribution estimate after selection | No |
| External PlantDoc TEST subset | Final external domain-shift benchmark after selection | No |

TEST results must not be retrospectively described as selection evidence.

## Domain shift and limitations

The large gap between the Dataset V2 INTERNAL TEST and the external PlantDoc
subset is evidence of substantial domain shift. PlantDoc imagery differs in
background, framing, lighting, capture conditions, symptom presentation, and
possibly cultivar or disease-stage distribution. The INTERNAL TEST result must
therefore not be marketed as real-world field accuracy.
It is not a field-accuracy claim.

The PlantDoc estimate is itself limited: it contains only 99 evaluated images,
has small per-class supports, covers 12 of 39 deployed classes, and excludes all
labels that could not be mapped with semantic certainty. It is valuable as a
limitation finding, not as a complete product-accuracy statement.

AgriDiagnose remains a decision-support system rather than a guaranteed
agricultural diagnosis. Results should be combined with local agronomic advice.

## Scientific integrity and no post-TEST tuning

Model V2 is frozen at the checksum recorded above. No post-TEST tuning is
allowed: no threshold, preprocessing, augmentation, architecture, split,
optimizer, or weights may be changed in response to either TEST result while
still calling the artifact Model V2. The current PlantDoc TEST observations must
not become a tuning target.

## Recommended Model V3 direction

A separately pre-declared Model V3 study may investigate more field-oriented
training data, diverse cameras/backgrounds/lighting, stronger but scientifically
fixed real-world augmentation, and improved out-of-distribution or uncertainty
handling. It should use an independent real-world development validation set and
reserve a **new untouched external benchmark** for its final evaluation. The
current PlantDoc TEST must remain final Model V2 evidence rather than be recycled
as Model V3 development data.

Machine-readable values and policy flags are frozen in
[`training/config/model-v2-final-selection.json`](../training/config/model-v2-final-selection.json).
