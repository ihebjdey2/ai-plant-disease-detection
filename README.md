# AgriDiagnose AI

<div align="center">

**AI-assisted plant disease analysis with a scientifically frozen 39-class MobileNetV2 model, Flask, TensorFlow, and a production-oriented application architecture.**

[![CI](https://github.com/ihebjdey2/ai-plant-disease-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/ihebjdey2/ai-plant-disease-detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.2.5-000000?logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)
![Model](https://img.shields.io/badge/Model-MobileNetV2-6C5CE7)
![Dataset](https://img.shields.io/badge/Dataset-73%2C563%20images-2E8B57)
![Classes](https://img.shields.io/badge/Classes-39-1F6FEB)
![Crops](https://img.shields.io/badge/Crops-14-228B22)
![Languages](https://img.shields.io/badge/UI-EN%20%7C%20FR%20%7C%20AR-8A2BE2)

</div>

> [!IMPORTANT]
> AgriDiagnose AI is a **decision-support system**, not a confirmed agricultural diagnosis tool. Treatment and crop-care decisions should be validated with qualified local agricultural expertise.

---

## Overview

AgriDiagnose AI is a full-stack machine-learning application for plant leaf analysis. It combines a frozen TensorFlow/Keras image classifier with a secure Flask web application, multilingual user experience, prediction history, crop-health dashboards, and a versioned REST API.

The project goes beyond a simple notebook demo: it includes reproducible dataset construction, group-aware data splitting, controlled model experiments, integrity-locked model artifacts, automated tests, database migrations, CI, external evaluation, and documented scientific limitations.

### At a glance

| Area | Current state |
| --- | --- |
| **Dataset V2** | **73,563 audited image files** |
| **Training split** | **58,857 images** |
| **Validation split** | **7,362 images** |
| **Internal TEST split** | **7,344 images** |
| **Output taxonomy** | **39 classes** |
| **Supported crops** | **14 crops** |
| **Frozen model** | MobileNetV2 / Experiment A |
| **Internal TEST accuracy** | **95.44%** |
| **Internal TEST Macro-F1** | **95.92%** |
| **External PlantDoc subset accuracy** | **41.41%** |
| **Application languages** | English, French, Arabic |
| **Deployment state** | Integrated as application default; not claimed as externally production-hosted |

---

## Dataset V2 — 73,563 audited images

The final Dataset V2 contains **73,563 audited image files**, well above the 50K-image scale, while preserving all **39 deployed classes**.

The split is group-aware and deterministic to reduce leakage between TRAIN, VALIDATION, and TEST.

| Partition | Images | Share | Groups | Class coverage |
| --- | ---: | ---: | ---: | ---: |
| **TRAIN** | **58,857** | 80.01% | 56,478 | 39/39 |
| **VALIDATION** | **7,362** | 10.01% | 7,098 | 39/39 |
| **INTERNAL TEST** | **7,344** | 9.98% | 7,087 | 39/39 |
| **Total** | **73,563** | **100%** | **70,663** | **39/39** |

### Dataset domains

| Domain | Images |
| --- | ---: |
| Historical / controlled | 55,423 |
| Real-world | 18,140 |
| **Total** | **73,563** |

Real-world data includes PLDD-UP, Seasonal Corn, PlantDoc TRAIN-source images, and Banu/Deb Potato images. The official **PlantDoc TEST split remains outside Dataset V2** and was reserved for external-domain evaluation.

### Leakage controls

The frozen split enforces:

- zero group leakage;
- zero known SHA-256 content leakage;
- zero PlantDoc TEST contamination;
- complete 39-class coverage across TRAIN, VALIDATION, and INTERNAL TEST;
- deterministic split generation using a fixed seed and group-aware allocation.

See [Dataset V2 group-aware split](docs/dataset-v2-group-aware-split.md) and [Dataset V2 39-class composition](docs/dataset-v2-39class-composition.md).

---

## Frozen Model V2

The selected application model is:

```text
models/agri-diagnose-v2-exp-a.keras
```

Model V2 / Experiment A was selected **strictly on VALIDATION performance before either final TEST was opened**.

| Property | Frozen contract |
| --- | --- |
| Architecture | ImageNet MobileNetV2 → GlobalAveragePooling2D → Dropout(0.25) → Dense(39, softmax) |
| Input | RGB `224 × 224 × 3` |
| Preprocessing | Resize → `float32` → divide by `255.0` |
| Inference augmentation | None |
| Outputs | 39 ordered classes |
| Background safety class | `Background without leaves`, index `4` |
| Confidence threshold | `60%`; exactly `60%` is accepted |
| Load mode | `compile=False` |
| Frozen SHA-256 | `bba4d044bcafbbee6dcd9f604e9c3f10c42f2531f17f21769b991540e36b8ca0` |

The frozen artifact is SHA-256 verified before its first cached load. A compatible custom model can still be configured through `MODEL_PATH`.

The legacy `plant_disease_model.h5` remains available for explicit rollback.

---

## Scientific evaluation

Experiment A was selected using **VALIDATION Macro-F1 only**. Experiment B was rejected and was never evaluated on either final TEST.

| Evaluation | Scope | Accuracy | Macro-F1 | Weighted-F1 | Role |
| --- | --- | ---: | ---: | ---: | --- |
| **VALIDATION** | 7,362 images · 39 classes | 95.48% | 96.01% | — | Candidate selection |
| **Dataset V2 INTERNAL TEST** | 7,344 images · 39 classes | **95.44%** | **95.92%** | **95.43%** | Final in-distribution estimate |
| **External PlantDoc TEST subset** | 99 images · 12 matched classes | **41.41%** | **45.79%** | **43.35%** | External domain-shift benchmark |

> [!NOTE]
> The **95.44% INTERNAL TEST accuracy is not a field-accuracy claim**.  
> The PlantDoc result is a conservative **99-image / 12-class semantic-overlap subset**, not a full 39-class PlantDoc benchmark. The gap between internal and external results demonstrates substantial domain shift.

Model V2 is scientifically **frozen**. Its final TEST observations must not be used for post-TEST tuning.

See [Model V2 final evaluation and provenance](docs/model-v2-final-evaluation.md).

---

## Supported crops and classes

The frozen taxonomy contains **38 plant-health classes across 14 crops**, plus one explicit `Background without leaves` safety class.

| Crop | Output classes | Count |
| --- | --- | ---: |
| Apple | Apple scab, Black rot, Cedar apple rust, healthy | 4 |
| Blueberry | healthy | 1 |
| Cherry | Powdery mildew, healthy | 2 |
| Corn | Cercospora leaf spot, Common rust, Northern Leaf Blight, healthy | 4 |
| Grape | Black rot, Esca, Leaf blight, healthy | 4 |
| Orange | Huanglongbing | 1 |
| Peach | Bacterial spot, healthy | 2 |
| Bell pepper | Bacterial spot, healthy | 2 |
| Potato | Early blight, Late blight, healthy | 3 |
| Raspberry | healthy | 1 |
| Soybean | healthy | 1 |
| Squash | Powdery mildew | 1 |
| Strawberry | Leaf scorch, healthy | 2 |
| Tomato | Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, mosaic virus, healthy | 10 |
| Non-leaf safety class | Background without leaves | 1 |
| **Total** | **Complete frozen taxonomy** | **39** |

Coverage describes the model's output taxonomy; it does not guarantee field accuracy for every crop variety, symptom stage, environment, or capture condition.

---

## Application features

### Plant analysis

- Image upload for JPG, JPEG, PNG, and WEBP files up to 5 MB
- Pillow validation of actual image contents
- MobileNetV2 inference with Top-3 confidence-ranked predictions
- Explicit `healthy`, `diseased`, `uncertain`, and `no_leaf` statuses
- Disease description, symptoms, causes, treatment, and prevention metadata
- Responsible-use warnings for uncertain and no-leaf predictions

### User experience

- Registration, login, and logout
- Flask-Login sessions
- Werkzeug password hashing
- Flask-WTF CSRF protection
- User-scoped prediction history
- Pagination, prediction details, delete, and clear-history actions
- Dashboard statistics and recent scans
- Responsive interface

### Multilingual UI

The full user-facing web interface supports:

- **English (`en`)**
- **French (`fr`)**
- **Arabic (`ar`)**

Arabic uses right-to-left layout, while language selection is persisted in the session and can fall back to the browser's supported language preference.

### Engineering

- Flask application factory
- Modular Blueprints
- Shared prediction service for web and REST API
- SQLAlchemy persistence
- SQLite for development
- PostgreSQL support
- Flask-Migrate / Alembic migrations
- Rotating application logs
- Friendly error handling
- Frozen model checksum verification
- Reproducible Kaggle experiment workflows
- Automated pytest suite
- GitHub Actions CI on Python 3.11

---

## Architecture

```text
                         ┌───────────────────────────┐
                         │       Browser / API       │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │       Flask Routes        │
                         │ auth · dashboard · API    │
                         └─────────────┬─────────────┘
                                       │
                     ┌─────────────────┴─────────────────┐
                     │                                   │
                     ▼                                   ▼
          ┌─────────────────────┐             ┌──────────────────────┐
          │ Prediction Service  │             │ SQLAlchemy / History │
          └──────────┬──────────┘             └──────────────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Frozen Model V2     │
          │ MobileNetV2 · 39    │
          └─────────────────────┘
```

Prediction status precedence is fixed:

1. class index `4` → `no_leaf`;
2. confidence below `60%` → `uncertain`;
3. accepted healthy class → `healthy`;
4. every other accepted prediction → `diseased`.

---

## Repository structure

```text
app/
├── data/                 structured disease metadata
├── models/               SQLAlchemy entities
├── routes/               auth, dashboard, prediction, history, REST API
├── services/             prediction, disease guidance, weather boundary
└── extensions.py         SQLAlchemy, Login, Migrate, CSRF

models/
└── agri-diagnose-v2-exp-a.keras

training/                 dataset manifests, policies, provenance, experiment tooling
evaluation/               offline evaluator and PlantDoc mapping
scripts/                  dataset, evaluation, and Kaggle utilities
notebooks/                controlled Kaggle Experiment A/B workflows
docs/                     scientific and engineering documentation
tests/                    application, policy, integration, and provenance tests
migrations/               Alembic migration history
config.py                 environment-based configuration
run.py                    development entry point
```

---

## Technology stack

| Area | Technology |
| --- | --- |
| Backend | Python 3.11, Flask 2.2.5, Jinja2 |
| Machine learning | TensorFlow 2.15.0, Keras 2.15.0, MobileNetV2 |
| Data / vision | NumPy, pandas, scikit-learn, Pillow, OpenCV |
| Persistence | Flask-SQLAlchemy, SQLite, PostgreSQL / psycopg |
| Authentication | Flask-Login, Werkzeug, Flask-WTF |
| Migrations | Flask-Migrate, Alembic |
| Testing | pytest, pytest-mock |
| CI | GitHub Actions |
| Frontend | HTML, CSS, JavaScript |

---

## Quick start

### Requirements

- Python `3.11`
- Git
- Enough disk space for TensorFlow and the included frozen model

### Windows PowerShell

```powershell
git clone https://github.com/ihebjdey2/ai-plant-disease-detection.git
cd ai-plant-disease-detection

py -3.11 -m venv C:\pdvenv
C:\pdvenv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

flask --app run.py db upgrade
python run.py
```

### Linux / macOS

```bash
git clone https://github.com/ihebjdey2/ai-plant-disease-detection.git
cd ai-plant-disease-detection

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

flask --app run.py db upgrade
python run.py
```

Open:

```text
http://127.0.0.1:5000
```

The built-in Flask development server is not intended for production hosting.

---

## Configuration

Development works without a `.env` file and falls back to SQLite.

| Variable | Purpose |
| --- | --- |
| `FLASK_ENV` | `development` by default; use `production` for production validation |
| `SECRET_KEY` | Required in production |
| `DATABASE_URL` | SQLAlchemy database URL; defaults to SQLite in development |
| `MODEL_PATH` | Optional custom compatible model; frozen Model V2 is the default |
| `UPLOAD_FOLDER` | Defaults to `static/uploads` |
| `PREDICTION_CONFIDENCE_THRESHOLD` | Defaults to `60` |
| `WEATHER_API_KEY` | Reserved for a future configured weather provider |

Example:

```env
FLASK_ENV=production
SECRET_KEY=replace-with-a-long-random-secret
DATABASE_URL=postgresql+psycopg://username:password@localhost:5432/agridiagnose
```

Production mode fails explicitly if required production environment variables are missing.

---

## REST API

### `POST /api/v1/predict`

Send one image as multipart form data:

```bash
curl -X POST http://127.0.0.1:5000/api/v1/predict \
  -F "image=@leaf.jpg"
```

Representative response:

```json
{
  "success": true,
  "prediction": {
    "class_index": 31,
    "disease": "Tomato Late blight",
    "confidence": 94.72,
    "status": "diseased",
    "is_background": false,
    "uncertain": false
  },
  "top_predictions": [
    {
      "class_index": 31,
      "disease": "Tomato Late blight",
      "confidence": 94.72
    }
  ]
}
```

Invalid or corrupted images return a friendly `400` response. Uploads larger than 5 MB are rejected.

> [!WARNING]
> The REST API is currently public and stateless. Add authentication, quotas, and rate limiting before exposing it as a public production API.

---

## Validation and CI

Install development dependencies and run:

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
pytest
flask --app run.py db current
flask --app run.py db check
git diff --check
```

GitHub Actions validates the project on Python 3.11.

The Model V2 integration checks include:

- committed model SHA-256;
- `compile=False` loading;
- `224 × 224 × 3` input contract;
- 39-value finite synthetic output;
- RGB / `float32` / `/255.0` preprocessing;
- confidence-threshold semantics;
- custom `MODEL_PATH` compatibility;
- no dependency on INTERNAL TEST or PlantDoc TEST images.

---

## Documentation

- [Model V2 final evaluation and provenance](docs/model-v2-final-evaluation.md)
- [Dataset V2 39-class composition](docs/dataset-v2-39class-composition.md)
- [Dataset V2 group-aware split](docs/dataset-v2-group-aware-split.md)
- [Frozen Model V2 baseline policy](training/config/model-v2-training-policy.json)
- [Frozen Experiment B augmentation policy](training/config/model-v2-experiment-b-policy.json)
- [Experiment A Kaggle workflow](docs/kaggle-model-v2-experiment-a.md)
- [Experiment B Kaggle workflow](docs/kaggle-model-v2-experiment-b.md)
- [Conservative PlantDoc semantic mapping](evaluation/mappings/plantdoc.json)
- [Frozen machine-readable selection metadata](training/config/model-v2-final-selection.json)

---

## Known limitations

- External PlantDoc performance reveals significant domain shift.
- The external benchmark covers only 12 of the 39 deployed classes.
- The PlantDoc subset does not evaluate the background/no-leaf safety class.
- Confidence is not a guarantee of correctness or calibration in every domain.
- Disease guidance is general and does not replace local agronomic expertise.
- Twenty-three deployed classes still lack real-world source data in Dataset V2.
- The REST API does not yet include authentication, quotas, or rate limiting.
- A production WSGI deployment profile and live weather provider are not yet included.
- Frozen Model V2 must not be tuned using its final TEST observations.

---

## Roadmap

### Model V2 application lifecycle

- [x] Build and audit Dataset V2
- [x] Train Experiment A
- [x] Train Experiment B
- [x] Select Experiment A on VALIDATION
- [x] Freeze Model V2
- [x] Run INTERNAL TEST
- [x] Run external PlantDoc TEST
- [x] Freeze evaluation provenance
- [x] Integrate the frozen Model V2 artifact as the application default
- [ ] Complete local functional QA
- [ ] Complete production-readiness review
- [ ] Deploy the application
- [ ] Establish monitoring and rollback procedures
- [ ] Publish a stable Model V2 release

### Future Model V3

Model V3 should focus primarily on **real-world generalization**, not merely increasing the already-strong internal score.

Planned research directions include:

- more field-oriented training data;
- broader camera, lighting, background, cultivar, and symptom-stage diversity;
- pre-declared real-world augmentation;
- stronger uncertainty and out-of-distribution handling;
- independent development validation for field imagery;
- a **new untouched external benchmark** reserved for final Model V3 evaluation.

The current PlantDoc TEST result remains frozen Model V2 evidence and must not become a Model V3 tuning target.

---

## Responsible use

AgriDiagnose AI is a software-engineering and machine-learning portfolio project designed to support observation and triage.

It should **not** replace:

- laboratory testing;
- agricultural extension services;
- professional agronomic diagnosis;
- crop-treatment decisions made by qualified specialists.

---

<div align="center">

### AgriDiagnose AI

**73,563 audited images · 58,857 training images · 39 classes · 14 crops · 3 languages**

Built with **Flask · TensorFlow · MobileNetV2 · PostgreSQL · GitHub Actions**

</div>
