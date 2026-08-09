# Dataset V2 Inventory

Audit date: 2026-08-09
Scope: public-source discovery and metadata review only. No dataset images were downloaded, no split was built, and no model or application code was changed.

## Decision summary

AgriDiagnose V2 should eventually combine a reproducible controlled baseline with independent field/smartphone images, but only after a source-level license, label, and duplicate audit. The best immediate candidates are PlantDoc **TRAIN**, the original images from the Seasonal Corn dataset, PLDD-UP, the original subset of the 804-image Potato Leaf Disease Dataset, and the labeled single-disease portion of Plant Pathology 2020 subject to Kaggle terms review.

PlantDoc **TEST remains permanently locked**. PlantWild is more valuable as an additional unseen-domain benchmark than as training data and its `CC BY-NC-ND 4.0` license is restrictive. PlantSeg should also remain reserved until its current release, split, and license metadata are reconciled.

No source in this inventory is approved for download or training yet.

## Current model taxonomy

The deployed order below comes directly from `app/services/prediction_service.py`. It is the fixed Model V2 target taxonomy.

| Index | Deployed class | Status |
|---:|---|---|
| 0 | Apple Apple scab | diseased |
| 1 | Apple Black rot | diseased |
| 2 | Apple Cedar apple rust | diseased |
| 3 | Apple healthy | healthy |
| 4 | Background without leaves | no-leaf |
| 5 | Blueberry healthy | healthy |
| 6 | Cherry Powdery mildew | diseased |
| 7 | Cherry healthy | healthy |
| 8 | Corn Cercospora leaf spot | diseased |
| 9 | Corn Common rust | diseased |
| 10 | Corn Northern Leaf Blight | diseased |
| 11 | Corn healthy | healthy |
| 12 | Grape Black rot | diseased |
| 13 | Grape Esca | diseased |
| 14 | Grape Leaf blight | diseased |
| 15 | Grape healthy | healthy |
| 16 | Orange Huanglongbing | diseased |
| 17 | Peach Bacterial spot | diseased |
| 18 | Peach healthy | healthy |
| 19 | Bell pepper Bacterial spot | diseased |
| 20 | Bell pepper healthy | healthy |
| 21 | Potato Early blight | diseased |
| 22 | Potato Late blight | diseased |
| 23 | Potato healthy | healthy |
| 24 | Raspberry healthy | healthy |
| 25 | Soybean healthy | healthy |
| 26 | Squash Powdery mildew | diseased |
| 27 | Strawberry Leaf scorch | diseased |
| 28 | Strawberry healthy | healthy |
| 29 | Tomato Bacterial spot | diseased |
| 30 | Tomato Early blight | diseased |
| 31 | Tomato Late blight | diseased |
| 32 | Tomato Leaf Mold | diseased |
| 33 | Tomato Septoria leaf spot | diseased |
| 34 | Tomato Spider mites | diseased |
| 35 | Tomato Target Spot | diseased |
| 36 | Tomato Yellow Leaf Curl Virus | diseased |
| 37 | Tomato mosaic virus | diseased |
| 38 | Tomato healthy | healthy |

Confirmed totals:

- 39 outputs: 26 diseased, 12 healthy, and 1 background/no-leaf class.
- `Background without leaves` is index 4.
- 14 crops: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
- Production preprocessing is RGB, resize to 224 × 224, `float32`, divide by `255.0`.

## Original training provenance audit

### Confirmed

- `model.ipynb` points to a local directory named `PlantVillage` on another Windows account. That path is not present in this workspace.
- Keras `ImageDataGenerator` used `rescale=1./255`, 224 × 224 RGB inputs, categorical labels, batch size 32, and `validation_split=0.2`.
- The notebook reports 44,371 training images and 11,077 validation images: 55,448 total across 39 directories.
- The same augmented generator was used for training and validation. Rotation, shifts, shear, zoom, and horizontal flip could therefore also affect validation samples.
- No independent test set is shown. The notebook's final example evaluates a batch from its validation generator.
- The network is ImageNet MobileNetV2 with average pooling, dense/dropout layers, and 39-way softmax.
- The H5 metadata confirms a 39-output Keras model and an Adam learning rate of `1e-5`; it does not contain a class manifest, source URL, file hashes, or per-class provenance.
- There is no original training-image manifest, checksum list, acquisition record, or source license in the repository.

### Likely

- The 38 plant classes are a direct semantic match to the canonical PlantVillage 38-class taxonomy. The original source was therefore very likely PlantVillage or a repackaging of it.
- The 55,448 total is close to the canonical 54,306 PlantVillage images plus a separate background class. This numerical similarity is evidence, not proof.
- The background/no-leaf data came from a different source because canonical PlantVillage has only 38 plant classes. Its origin is unknown.

### Unknown

- Which PlantVillage release or mirror was used: author GitHub, Mendeley, Kaggle, or another repackaging.
- Whether the 55,448 files were all originals, included offline augmentations, or included duplicate/resized copies. On-the-fly augmentation is confirmed; offline augmentation is not.
- The exact background class image count, source, license, and capture distribution.
- Whether images of the same physical leaf crossed the notebook's random 80/20 directory split.
- Whether the intended fine-tuning actually unfroze later MobileNetV2 layers; the shown code keeps the base model frozen before changing only the first 100 layers to `trainable=False`.
- Exact reproducibility: no generator seed, file manifest, environment lock, or dataset hash was recorded.

The original validation score must therefore be treated as an internal, controlled-domain estimate, not a reproducible external performance claim.

## Locked benchmarks

### PlantDoc TEST — locked now

- Official source: [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset), Singh et al. (2020), CC BY 4.0.
- The current repository-tree audit observes 236 TEST images in 27 directories. The existing exact mapping evaluates 99 images across 12 deployed classes.
- Current baseline: 20.20% accuracy, 23.44% macro F1, 23.04% weighted F1, 51 uncertain predictions, and 15 false no-leaf predictions.
- Never use this split for training, validation, augmentation, threshold selection, class balancing, or model selection.

The PlantDoc paper describes 2,598 images, while the current official Git tree contains 2,578 image paths (2,342 TRAIN and 236 TEST). This difference must be recorded rather than silently normalized.

### Benchmark candidates — not locked yet

- **PlantWild v1 TEST:** strong in-the-wild stress test. The authors' loader creates a reproducible split and the community metadata reports 3,677 test images. Keep it unseen if legal approval is obtained. Do not use it for training under the present `CC BY-NC-ND 4.0` terms.
- **PlantSeg official TEST:** potentially valuable for lesion-heavy field imagery, but it is primarily a segmentation dataset. The latest Zenodo release and the published paper expose inconsistent image/version metadata, and the Zenodo Rights field is blank while the paper reports CC BY-NC 4.0. Reserve pending clarification.
- **Plant Pathology 2021:** natural apple imagery with multi-label combinations. It could be a strong apple benchmark, but an immutable local subset and legal terms must be approved before any data is consumed.

## Candidate public datasets

Visual/provenance categories:

- **A:** likely original controlled PlantVillage-style data
- **B:** repackaged or augmented PlantVillage-style data
- **C:** independent real-world/field data
- **D:** mixed provenance
- **E:** unknown provenance

Training suitability is a metadata/legal classification, not legal advice. `REQUIRES_ATTRIBUTION` still requires preserving the dataset citation and license notice in future manifests and distributions.

| Dataset | Original source | Images / classes | Category / domain | Relevant crops | Exact deployed overlaps | License / suitability | PlantVillage-derived? | Suggested role | Priority |
|---|---|---:|---|---|---:|---|---|---|---|
| PlantDoc | [authors' GitHub](https://github.com/pratikkayal/PlantDoc-Dataset) | paper 2,598; Git tree 2,578 / 27–28 dirs | C / internet real-world | 13 deployed crops except Orange | 13 in TRAIN; 12 in TEST | CC BY 4.0 / REQUIRES_ATTRIBUTION | No known derivation; web-source duplicate risk | TRAIN exact subset; TEST locked | A |
| Seasonal Corn Leaf Disease Dataset | [Mendeley DOI](https://doi.org/10.17632/vy629dngm8.1) | 2,943 originals + 7,500 augmentations / 5 | C / multi-farm field | Corn | 3 | CC BY 4.0 / REQUIRES_ATTRIBUTION | No | TRAIN originals only | A |
| PLDD-UP | [Mendeley DOI](https://doi.org/10.17632/3j4nfkvp2n.1) | 15,519 / 3 | C / operational fields, cameras + phones | Potato | 3 | CC BY 4.0 / REQUIRES_ATTRIBUTION | No stated derivation | TRAIN candidate | A |
| Potato Leaf Disease Dataset | [Mendeley DOI](https://doi.org/10.17632/d5b3fzpw3g.1) | 804 originals; 2,400 augmented / 6 | C / field, iPhone 15 | Potato | 2 | CC BY 4.0 / REQUIRES_ATTRIBUTION | No stated derivation | TRAIN originals only | A |
| Plant Pathology 2020 FGVC7 | [official Kaggle competition](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7) / [paper](https://doi.org/10.1111/tpj.14838) | 3,651 total; 1,821 labeled competition train / 4 | C / orchard, DSLR + phones | Apple | 3 | Kaggle terms not captured as an open license / UNCLEAR | No | conditional TRAIN; exclude multiple-disease | A |
| PlantWild v1/v2 | [authors' repository](https://github.com/tqwei05/MVPDR) / [official HF](https://huggingface.co/datasets/uqtwei2/PlantWild) | 18,542 / 89 (v1); 11,488 / 115 (v2) | C / in-the-wild web imagery | all except explicit Orange; generic citrus present | 17 disease labels confidently exact in v1; similar v2 | CC BY-NC-ND 4.0 / RESTRICTED | No; internet duplicate risk | reserved external benchmark | A |
| PlantSeg | [latest Zenodo](https://doi.org/10.5281/zenodo.17719108) / [paper](https://arxiv.org/abs/2409.04038) | release/paper vary; 115 diseases | C / in-the-wild with masks | many deployed crops | 18 disease labels confidently exact | Zenodo Rights blank; paper says CC BY-NC 4.0 / UNCLEAR-RESTRICTED | No; internet duplicate risk | reserve; legal/version audit | B |
| Plant Pathology 2021 FGVC8 | [official Kaggle competition](https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8) | 18,632 / 6 base labels, 12 combinations | C / field apple | Apple | Apple scab and healthy; rust is not proven cedar-specific | Kaggle terms not captured as an open license / UNCLEAR | No | reserve apple benchmark | B |
| Auburn Soybean Disease Image Dataset | [Dryad DOI](https://doi.org/10.5061/dryad.41ns1rnj3) | 9,981 / 8 | C / Alabama fields + controlled backgrounds | Soybean | 1 | CC0 under Dryad terms / SAFE_FOR_TRAINING | No | TRAIN healthy subset candidate | B |
| Citrus leaf/fruit disease dataset (Rauf et al.) | [Mendeley DOI](https://doi.org/10.17632/3f83gxmv57.1) | 759 / 5 leaf diseases/health + fruit | C / Pakistan orchards | Citrus; Orange not explicit | 0 exact; Citrus greening is a potential Orange HLB match | CC BY 4.0 / REQUIRES_ATTRIBUTION | No | semantic/species audit | B |
| CornLeafDiseaseCollection | [Mendeley DOI](https://doi.org/10.17632/w56xxnykcc.1) | 1,079 / 5 | C / Ecuador plantations, standardized smartphone capture | Corn | 1 (healthy) | CC BY 4.0 / REQUIRES_ATTRIBUTION | No | small controlled-field supplement | B |
| Pota-Toma-To | [Mendeley DOI](https://doi.org/10.17632/354fsxwccb.1) | 435 raw plus processed copies / 6 crop-disease combinations | E/D / acquisition not documented | Potato, Tomato | up to 6 potential; exact counts unknown | CC BY 4.0 / REQUIRES_ATTRIBUTION | Unknown | provenance + duplicate audit | B |
| Official PlantVillage | [authors' GitHub](https://github.com/spMohanty/PlantVillage-Dataset) / [HF card](https://huggingface.co/datasets/mohanty/PlantVillage) | 54,306 / 38 | A / controlled | all 14 crops | all 38 non-background classes | CC BY-SA 3.0 / REQUIRES_ATTRIBUTION + share-alike | It is the upstream source | controlled baseline reconstruction only | C |
| Corn Leaf Disease Classification Dataset | [Mendeley DOI](https://doi.org/10.17632/hmkd6nbngr.1) | ~17,000 / 5 | D / field plus public repositories | Corn | 3 possible | CC BY 4.0 / REQUIRES_ATTRIBUTION | Unknown mixture | dedup/provenance audit only | C |
| Multi-Crop Leaf Disease Dataset | [Mendeley DOI](https://doi.org/10.17632/z6jp232g5j.1) | 6,895 / unspecified label detail | E / unspecified | Corn, Potato, Tomato | unknown | CC BY 4.0 but source rights/provenance unclear / NOT_RECOMMENDED | Unknown | reject pending upstream provenance | REJECT |
| New Plant Diseases Dataset (Augmented) | [Kaggle page](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) | ~87,000 / 38 | B / augmented PlantVillage | all 14 crops | 38 but not independent | Kaggle page terms must be rechecked / NOT_RECOMMENDED | Yes | reject as new diversity | REJECT |
| LemonLENS | [Mendeley DOI](https://doi.org/10.17632/ccwh64fr2r.2) | 4,500 / 9 | C / smartphone/natural | Lemon, not Orange | 0 | CC BY 4.0 / REQUIRES_ATTRIBUTION | No stated derivation | reject for current taxonomy | REJECT |

## Priority A analysis and class overlap

### 1. PlantDoc TRAIN

The official repository tree contains 2,342 TRAIN image paths in 28 directories. Reusing the conservative mapping rules already committed for the locked test benchmark yields 13 exact target classes and 1,085 candidate images:

| Source label | Target class | Match | TRAIN images |
|---|---|---|---:|
| Apple Scab Leaf | Apple Apple scab | EXACT_MATCH | 83 |
| Corn Gray leaf spot | Corn Cercospora leaf spot | EXACT_MATCH | 64 |
| Potato leaf early blight | Potato Early blight | EXACT_MATCH | 109 |
| Potato leaf late blight | Potato Late blight | EXACT_MATCH | 97 |
| Squash Powdery mildew leaf | Squash Powdery mildew | EXACT_MATCH | 124 |
| Tomato Early blight leaf | Tomato Early blight | EXACT_MATCH | 79 |
| Tomato Septoria leaf spot | Tomato Septoria leaf spot | EXACT_MATCH | 140 |
| Tomato leaf bacterial spot | Tomato Bacterial spot | EXACT_MATCH | 101 |
| Tomato leaf late blight | Tomato Late blight | EXACT_MATCH | 101 |
| Tomato leaf mosaic virus | Tomato mosaic virus | EXACT_MATCH | 44 |
| Tomato mold leaf | Tomato Leaf Mold | EXACT_MATCH | 85 |
| Tomato two spotted spider mites leaf | Tomato Spider mites | EXACT_MATCH | 2 |
| grape leaf black rot | Grape Black rot | EXACT_MATCH | 56 |

Generic `* leaf` labels remain AMBIGUOUS because they do not explicitly state healthy status. Apple rust, corn blight, corn rust, bell-pepper leaf spot, and tomato yellow virus also remain AMBIGUOUS because the deployed diseases are more specific. The two spider-mite files are semantically exact but too small to be useful without another source.

PlantDoc images came from public web search, so Step 5B must hash and perceptually compare TRAIN against the locked TEST and every other acquired source before any split is designed.

### 2. Seasonal Corn Leaf Disease Dataset

The dataset was captured at several farms in Gurudaspur, Natore, Bangladesh with expert collaboration and includes location/environment metadata.

| Source label | Target class | Match | Original images |
|---|---|---|---:|
| Healthy | Corn healthy | EXACT_MATCH | 1,038 |
| Common_rust | Corn Common rust | EXACT_MATCH | 129 |
| Gray_leaf_spot | Corn Cercospora leaf spot | EXACT_MATCH | 1,497 |
| Bacterial Leaf Streak | — | NOT_SUPPORTED | 190 |
| Maize Chlorotic Mottle Virus | — | NOT_SUPPORTED | 89 |

Only the 2,943 originals may enter the next audit. The 7,500 generated variants must be ignored to avoid fake diversity and split leakage.

### 3. PLDD-UP

This 2026 release contains 15,519 high-resolution field images captured with a Nikon D5300 and smartphones under natural light in three Uttar Pradesh regions.

| Source label | Target class | Match | Images |
|---|---|---|---:|
| EB | Potato Early blight | EXACT_MATCH | 4,803 |
| LB | Potato Late blight | EXACT_MATCH | 6,116 |
| Healthy | Potato healthy | EXACT_MATCH | 4,600 |

It offers the largest clearly aligned field source found in this audit. Step 5B must still verify pathology documentation, near-duplicate bursts, location/date grouping, and whether frames of the same plant can cross future splits.

### 4. Potato Leaf Disease Dataset (Banu and Deb, 2026)

The 804 originals were captured at BARI in Bangladesh using an iPhone 15 under natural conditions. Exact class counts were not published on the source page.

| Source label | Target class | Match |
|---|---|---|
| Healthy | Potato healthy | EXACT_MATCH |
| Fungal Late Blight | Potato Late blight | EXACT_MATCH |
| Bacterial Soft Rot | — | NOT_SUPPORTED |
| Viral Leaf Roll | — | NOT_SUPPORTED |
| Viral PVX | — | NOT_SUPPORTED |
| Viral PVY | — | NOT_SUPPORTED |

Use only the original files. The 2,400-image augmented collection must not be treated as independent observations.

### 5. Plant Pathology 2020 FGVC7

The paper documents orchard images taken with a Canon DSLR and smartphones under variable illumination, angles, surfaces, and noise. The public labeled competition TRAIN has 1,821 images: 592 scab, 622 rust, 516 healthy, and 91 multiple-disease images.

| Source label | Target class | Match | Labeled TRAIN images |
|---|---|---|---:|
| scab | Apple Apple scab | EXACT_MATCH | 592 |
| rust | Apple Cedar apple rust | EXACT_MATCH | 622 |
| healthy | Apple healthy | EXACT_MATCH | 516 |
| multiple_diseases | — | NOT_SUPPORTED | 91 |

The paper establishes that this challenge's rust is cedar apple rust, so this is not a generic-text guess. The exact single-label candidate total is 1,730. Training approval remains conditional because the Kaggle competition terms were not available as a clear reusable open-data license during this audit.

### 6. PlantWild — reserve, do not train

PlantWild v1 contains 18,542 web-sourced in-the-wild images in 89 classes; v2 adds/refines disease classes. The official Hugging Face record states `CC BY-NC-ND 4.0`, which is non-commercial and prohibits derivatives. It is therefore classified RESTRICTED.

Confident disease-label overlaps in the v1 label metadata are:

`Apple Black rot`, `Apple Apple scab`, `Cherry Powdery mildew`, `Corn Cercospora leaf spot`, `Corn Northern Leaf Blight`, `Grape Black rot`, `Potato Early blight`, `Potato Late blight`, `Squash Powdery mildew`, `Strawberry Leaf scorch`, `Tomato Bacterial spot`, `Tomato Early blight`, `Tomato Late blight`, `Tomato Leaf Mold`, `Tomato Septoria leaf spot`, `Tomato Yellow Leaf Curl Virus`, and `Tomato mosaic virus`.

`Bell pepper leaf spot` is a POTENTIAL_MATCH to bacterial spot but is not accepted without source-level label documentation. `Apple rust`, `Corn rust`, and `Grape leaf spot` are AMBIGUOUS because the deployed labels are more specific. Generic crop `leaf` labels are also AMBIGUOUS rather than silently mapped to healthy. `Citrus greening disease` is a POTENTIAL_MATCH to Orange Huanglongbing because the source crop is generic citrus.

Because it is unusually valuable as an unseen-domain challenge and legally restricted for normal product training, the recommended role is a locked benchmark after human/legal approval.

## Priority B analysis

### PlantSeg

The official class list includes exact disease matches for Apple black rot/scab, Bell pepper bacterial spot, Cherry powdery mildew, Corn gray leaf spot/Northern leaf blight, Grape black rot, Potato early/late blight, Squash powdery mildew, Strawberry leaf scorch, and the same seven supported Tomato diseases listed for PlantWild. Generic apple/corn rust and grape leaf spot remain AMBIGUOUS. Citrus greening is a POTENTIAL_MATCH because the host is not explicitly Orange.

PlantSeg is disease-segmentation data, not a drop-in 39-class image folder. Its version history has changed file sizes and counts. The latest Zenodo page leaves the license field blank, while the later paper states CC BY-NC 4.0. These must be resolved before download. If approved, preserve its official test split and consider only the official train split in a later design.

### Plant Pathology 2021 FGVC8

The 18,632 natural apple images include scab, healthy, rust, powdery mildew, frog-eye leaf spot, complex, and multi-label combinations. Only pure `scab` and `healthy` are currently exact. `rust` is AMBIGUOUS because the 2021 label does not itself prove cedar apple rust, and every multi-label combination is NOT_SUPPORTED by the current single-label taxonomy. Reserve this source for an apple-domain benchmark unless Step 5B establishes a clean, legally permitted use.

### Auburn Soybean Disease Image Dataset (ASDID)

ASDID contains 9,981 raw originals from Alabama field seasons in 2020–2021, captured with a DSLR and a Motorola phone. It mixes attached leaves in canopy with detached leaves on grass or white surfaces. Only `healthy` maps exactly to `Soybean healthy`; its seven disease/deficiency labels are not deployed. Dryad publishes datasets under CC0 and scholarly citation should still be retained. Its 43.36 GB size and one-class overlap make it lower priority despite excellent diversity.

### Citrus leaf/fruit disease dataset

Rauf et al. provide 759 citrus images from Pakistan and labels including greening, canker, black spot, melanose, and healthy. `Citrus greening` is biologically associated with Huanglongbing, but the deployed crop is specifically Orange while the source page says Citrus. It remains POTENTIAL_MATCH until the underlying species and leaf-only subset are confirmed. Fruit images must not be mixed into a leaf classifier without a deliberate design decision.

### CornLeafDiseaseCollection

The 1,079 images were collected in Ecuador plantations with a smartphone at a standardized distance. `healthy` is an exact match. The source describes `common rust (Southern rust)`, which is internally inconsistent with the deployed `Corn Common rust`; it is AMBIGUOUS and must not be forced. Other diseases are not supported.

### Pota-Toma-To

The source offers 435 raw Potato/Tomato images for Early Blight, Late Blight, and Healthy, plus processed copies. Label names potentially align with six deployed crop-class combinations. However, acquisition location, devices, class counts, upstream sources, and duplication are not documented on the landing page. Keep it at Priority B until the raw files and database description can be inspected. Never combine the processed copies with their originals as independent samples.

## Priority C and rejection rationale

- **Official PlantVillage — Priority C for diversity:** canonical and well documented, but controlled and probably the source family already used. It is valuable to reconstruct a reproducible controlled baseline, not to claim a new domain.
- **17k Corn classification dataset — Priority C:** claims field diversity but explicitly mixes field observations and public repositories. Source-level provenance and duplicate relationships are unknown.
- **New Plant Diseases Dataset (Augmented) — REJECT as new diversity:** it is an augmented/repackaged PlantVillage 38-class collection. It creates volume, not independent plants or environments.
- **Multi-Crop Leaf Disease Dataset — REJECT pending provenance:** no sufficiently detailed label/acquisition/upstream record was found. A CC BY landing-page declaration cannot by itself prove rights to unidentified upstream images.
- **LemonLENS and other lemon-only collections — REJECT for the fixed taxonomy:** potentially good field imagery but Lemon is not Orange. Cross-crop mapping would violate the current taxonomy.
- Community Kaggle mirrors and small derivative “field plant disease” mixtures were not promoted when an upstream source was identifiable. Reuploads increase attribution, leakage, and duplicate risk.

## Crop coverage search record

All 14 deployed crops were explicitly checked. Apple has dedicated FGVC sources; Corn and Potato have strong field releases; Soybean has ASDID; Orange has only generic-citrus potential matches. Blueberry, Cherry, Grape, Peach, Bell pepper, Raspberry, Squash, Strawberry, and Tomato appear in PlantDoc and/or PlantWild/PlantSeg, but no additional independently acquired, clearly licensed, exact-label source was promoted above Priority B during this pass.

The absence of a promoted dataset does not mean no public images exist. It means no source reviewed here simultaneously met the crop/disease semantics, provenance, license, and real-world-diversity bar.

## Recommended source strategy

### Approve for Step 5B metadata/file audit — not training yet

1. PlantDoc TRAIN exact subset; keep TEST locked and run cross-split hashes first.
2. Seasonal Corn originals only; ignore generated augmentations.
3. PLDD-UP; verify labels and capture-group metadata before splitting.
4. Potato Leaf Disease Dataset originals only.
5. Plant Pathology 2020 labeled single-disease images, only after Kaggle terms approval.
6. Canonical PlantVillage only to reconstruct controlled provenance and leaf-group-aware splits.

### Keep or evaluate as external benchmarks

1. PlantDoc TEST — already locked.
2. PlantWild v1 TEST — preferred new real-world benchmark, pending restricted-license approval.
3. PlantSeg TEST — reserve pending version/license resolution and a classification adapter design.
4. Plant Pathology 2021 — reserve as an apple-domain/multi-label stress test pending terms review.

### Reject from Dataset V2 at this stage

1. Augmented PlantVillage/Kaggle mirrors as “new” diversity.
2. Multi-source datasets without traceable upstream provenance.
3. Processed/augmented copies when the corresponding originals exist.
4. Generic labels mapped to specific diseases without pathology evidence.
5. Lemon images mapped to Orange, or multi-disease images forced into a single deployed class.

## Quantitative scope and limitations

Across four candidates with published or locally audited class counts—PlantDoc TRAIN, Seasonal Corn originals, PLDD-UP, and Plant Pathology 2020 labeled TRAIN—there are at least **20,998 images attached to exact deployed classes before duplicate, corruption, license, and quality filtering**. This is not an approved or final dataset size. The exact usable count may be substantially lower, and large class/crop imbalance is expected.

The main expected gain is independent visual diversity: whole plants or multiple leaves, cluttered field/orchard backgrounds, different countries, cameras, times of day, lighting, angles, symptom stages, and capture distances. No accuracy improvement is claimed. Only a future leakage-safe experiment can measure whether these sources improve generalization.

## Human approvals required before Step 5B

1. Approve the shortlist and storage budget for metadata-first downloads.
2. Decide whether AgriDiagnose has commercial aspirations; this directly affects PlantWild and PlantSeg eligibility.
3. Approve legal review of Kaggle competition terms for Plant Pathology 2020/2021.
4. Decide whether PlantWild v1 TEST and PlantSeg TEST should be formally locked before any training-source download.
5. Approve canonical PlantVillage reconstruction and compliance with CC BY-SA 3.0 attribution/share-alike obligations.
6. Decide whether the unknown background/no-leaf provenance should trigger a separate source-discovery audit.
7. Approve a strict Step 5B rule: source manifests and hashes first, then semantic review, then split design—never the reverse.

## Source notes

- Mohanty, Hughes, and Salathé, [PlantVillage repository](https://github.com/spMohanty/PlantVillage-Dataset) and [2016 paper](https://doi.org/10.3389/fpls.2016.01419).
- Singh et al., [PlantDoc repository](https://github.com/pratikkayal/PlantDoc-Dataset) and [paper](https://arxiv.org/abs/1911.10317).
- Wei et al., [PlantWild project](https://github.com/tqwei05/MVPDR) and [paper](https://arxiv.org/abs/2408.03120).
- Wei et al., [PlantSeg latest Zenodo record](https://doi.org/10.5281/zenodo.17719108) and [paper](https://arxiv.org/abs/2409.04038).
- Thapa et al., [Plant Pathology 2020 paper](https://doi.org/10.1111/tpj.14838) and [competition](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7).
- [Plant Pathology 2021 competition](https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8).
- Ahmad, [Seasonal Corn dataset](https://doi.org/10.17632/vy629dngm8.1) (2025).
- Singh et al., [PLDD-UP](https://doi.org/10.17632/3j4nfkvp2n.1) (2026).
- Banu and Deb, [Potato Leaf Disease Dataset](https://doi.org/10.17632/d5b3fzpw3g.1) (2026).
- Bevers, Sikora, and Hardy, [ASDID](https://doi.org/10.5061/dryad.41ns1rnj3) (2022).
- Rauf et al., [Citrus disease dataset](https://doi.org/10.17632/3f83gxmv57.1) (2019).
