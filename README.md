# AgriDiagnose AI

[![CI](https://github.com/ihebjdey2/ai-plant-disease-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/ihebjdey2/ai-plant-disease-detection/actions/workflows/ci.yml)

AgriDiagnose AI is a Flask and TensorFlow application for AI-assisted plant
leaf analysis. It combines a scientifically frozen 39-class MobileNetV2 model
with secure accounts, a crop-health dashboard, user-scoped prediction history,
and a versioned REST API.

> [!IMPORTANT]
> Predictions are decision support, not confirmed agricultural diagnoses.
> Confirm crop-care and treatment decisions with qualified local expertise.

## Project status

- **Application model:** frozen Model V2 / Experiment A
- **Deployment state:** integrated as the application default
- **Model integrity:** SHA-256 verified before the first cached load
- **Legacy rollback:** `plant_disease_model.h5` remains available
- **Scientific state:** Model V2 is frozen; post-TEST tuning is prohibited
- **Quality gate:** Python 3.11 CI, automated tests, and Alembic migration checks

The default artifact is `models/agri-diagnose-v2-exp-a.keras`. A compatible
custom model can still be selected with `MODEL_PATH`.

## What the application provides

### Web application

- Registration, login, and logout with Werkzeug password hashing
- Flask-Login session authentication and Flask-WTF CSRF protection
- Responsive upload interface with image preview
- Modern responsive dashboard with explicit model-coverage cards
- Complete English, French, and Arabic web interface
- Persistent language selector, browser-language detection, and Arabic RTL layout
- JPG, JPEG, PNG, and WEBP uploads up to 5 MB
- Pillow validation of actual image content and UUID-based filenames
- Primary prediction with confidence and explicit status handling
- Explicit `healthy`, `diseased`, `uncertain`, and `no_leaf` statuses
- Dashboard totals, average confidence, recent scans, and frequent diseases
- User-scoped history with details, pagination, deletion, and clear-all actions
- Status-specific uncertainty/no-leaf warnings and a responsible-use disclaimer

### Languages and complete frontend coverage

Every user-facing web page is available in **English (`en`)**, **French
(`fr`)**, and **Arabic (`ar`)**. This includes registration, login, dashboard,
image upload, loading state, validation and error messages, statistics, scan
history, prediction details, status labels, dates, disease labels, and the
responsible-use notices. The selected language is stored in the user session;
when no preference has been saved, the application uses the browser's preferred
supported language. Arabic pages use right-to-left layout through the document
`dir="rtl"` attribute and responsive RTL-aware CSS.

The dashboard exposes the model's complete plant coverage rather than implying
that it can analyze every crop. The frozen output taxonomy contains **39 total
classes**: **38 plant-health classes across 14 crops**, plus one explicit
`Background without leaves` safety class. The interface derives the class count
for each crop directly from the authoritative taxonomy and translates every
displayed crop and class name without changing the stable English values stored
in the database or returned by the API.

| Supported crop | Model output classes | Count |
| --- | --- | ---: |
| Apple | Apple scab; Black rot; Cedar apple rust; healthy | 4 |
| Blueberry | healthy | 1 |
| Cherry | Powdery mildew; healthy | 2 |
| Corn | Cercospora leaf spot; Common rust; Northern Leaf Blight; healthy | 4 |
| Grape | Black rot; Esca; Leaf blight; healthy | 4 |
| Orange | Huanglongbing | 1 |
| Peach | Bacterial spot; healthy | 2 |
| Bell pepper | Bacterial spot; healthy | 2 |
| Potato | Early blight; Late blight; healthy | 3 |
| Raspberry | healthy | 1 |
| Soybean | healthy | 1 |
| Squash | Powdery mildew | 1 |
| Strawberry | Leaf scorch; healthy | 2 |
| Tomato | Bacterial spot; Early blight; Late blight; Leaf Mold; Septoria leaf spot; Spider mites; Target Spot; Yellow Leaf Curl Virus; mosaic virus; healthy | 10 |
| Non-leaf safety class | Background without leaves | 1 |
| **Total** | **Complete frozen output taxonomy** | **39** |

The localized dashboard therefore shows exactly these 14 supported crops:
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell pepper, Potato,
Raspberry, Soybean, Squash, Strawberry, and Tomato. This coverage statement is
about model outputs, not a guarantee of field accuracy or support for other
species, varieties, symptoms, or growing conditions.

### Engineering and data

- Flask application factory and modular Blueprints
- Shared prediction service for the web interface and REST API
- Confidence-ranked Top-3 results from the prediction engine and REST API
- SQLAlchemy persistence with SQLite for development and PostgreSQL support
- Flask-Migrate/Alembic database migrations
- Rotating application logs and friendly error pages
- Frozen taxonomy, model checksum, experiment policies, and evaluation provenance
- Reproducible Kaggle Experiment A/B workflows and offline evaluation tooling
- GitHub Actions validation on Python 3.11

Weather data is not fabricated: the weather boundary reports unavailable until
a real provider is implemented and configured.

## Scientific evaluation

Experiment A was selected strictly on **VALIDATION Macro-F1**, before the
Dataset V2 INTERNAL TEST or external PlantDoc TEST was opened. Experiment B
was rejected and was never evaluated on either TEST.

| Partition | Scope | Accuracy | Macro-F1 | Weighted-F1 | Role |
| --- | --- | ---: | ---: | ---: | --- |
| VALIDATION | 7,362 images, 39 classes | 95.48% | 96.01% | — | Candidate selection |
| Dataset V2 INTERNAL TEST | 7,344 images, 39 classes | 95.44% | 95.92% | 95.43% | Final in-distribution estimate |
| External PlantDoc TEST — conservative semantic-overlap subset | 99 images, 12 matched classes | 41.41% | 45.79% | 43.35% | Final domain-shift benchmark |

The INTERNAL TEST result measures performance within the Dataset V2
distribution; **it is not field accuracy**. The PlantDoc figure covers only a
conservative 99-image / 12-class semantic-overlap subset, not all PlantDoc
labels and not all 39 deployed classes. The external result demonstrates a
substantial external-domain shift between Dataset V2 and the more realistic
PlantDoc imagery.

See the [frozen Model V2 evaluation and provenance](docs/model-v2-final-evaluation.md)
for exact values, selection chronology, checksums, limitations, and the
no-post-TEST-tuning policy.

## Model contract

| Property | Frozen application contract |
| --- | --- |
| Architecture | ImageNet MobileNetV2 → GlobalAveragePooling2D → Dropout(0.25) → Dense(39, softmax) |
| Input | RGB image, `224 × 224 × 3` |
| Preprocessing | Resize, convert to `float32`, divide by `255.0` |
| Inference augmentation | None |
| Output taxonomy | 39 ordered classes |
| Background class | `Background without leaves`, index `4` |
| Default confidence threshold | `60%`; exactly `60%` is accepted |
| Frozen SHA-256 | `bba4d044bcafbbee6dcd9f604e9c3f10c42f2531f17f21769b991540e36b8ca0` |

The application deliberately does not use
`MobileNetV2.preprocess_input`. The authoritative class order lives in
`training/taxonomy.py`, is re-exported through `app/taxonomy.py`, and is
regression-tested across application, training, and evaluation code.

Prediction status precedence is fixed:

1. class index `4` → `no_leaf`;
2. confidence below the configured threshold (`60%` by default) → `uncertain`;
3. an accepted healthy class → `healthy`;
4. every other accepted prediction → `diseased`.

The default artifact is loaded with `compile=False`, cached once per process,
and checked for its input shape, output count, and frozen SHA-256. Distinct
custom `MODEL_PATH` artifacts are not forced to match the frozen checksum, but
must still satisfy the input/output contract.

## Architecture

```text
app/
├── data/                 structured disease metadata
├── models/               User and Prediction SQLAlchemy entities
├── routes/               auth, dashboard, prediction, history, REST API
├── services/             prediction, disease guidance, weather boundary
└── extensions.py         SQLAlchemy, Login, Migrate, CSRF
models/
└── agri-diagnose-v2-exp-a.keras
migrations/               Alembic migration history
templates/                Jinja pages
static/                   CSS, JavaScript, and runtime uploads
training/                 frozen policies, provenance, experiment tooling
evaluation/               offline evaluator and PlantDoc mapping
scripts/                  dataset, evaluation, and Kaggle utilities
notebooks/                controlled Kaggle Experiment A/B workflows
docs/                     dataset, training, and final-evaluation records
tests/                    application, ML-policy, and provenance tests
config.py                 environment-based configuration
run.py                    development entry point
```

The active request flow is intentionally simple:

```text
Browser or REST client
        ↓
Flask Blueprint → image validation → shared prediction service
        ↓                              ↓
SQLAlchemy history (web only)     cached frozen Model V2
```

The REST API is stateless and does not write prediction history. Authenticated
web scans are persisted only for the current user.

## Technology stack

| Area | Technology |
| --- | --- |
| Backend | Python 3.11, Flask 2.2.5, Jinja2 |
| Machine learning | TensorFlow 2.15.0, Keras 2.15.0, MobileNetV2 |
| Data | NumPy, pandas, scikit-learn, Pillow, OpenCV |
| Persistence | Flask-SQLAlchemy, SQLite, PostgreSQL/psycopg |
| Authentication | Flask-Login, Werkzeug password hashing, Flask-WTF |
| Migrations | Flask-Migrate, Alembic |
| Testing | pytest, pytest-mock, GitHub Actions |
| Frontend | HTML, CSS, JavaScript |

## Quick start

### Requirements

- Python `3.11`
- Git
- Enough disk space for TensorFlow and the included model

On Windows, a short virtual-environment path avoids TensorFlow installation
failures caused by the legacy path-length limit.

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

### Linux or macOS

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

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The built-in Flask server
is for local development, not production hosting.

## Configuration

The default development configuration needs no `.env` and uses SQLite. Copy
`.env.example` only when you are ready to replace its example values; never
commit real secrets. In particular, set or remove its example `DATABASE_URL`
before running migrations.

| Variable | Purpose and default |
| --- | --- |
| `FLASK_ENV` | `development` by default; use `production` to enable production validation |
| `SECRET_KEY` | Required in production; development generates a temporary key with a warning |
| `DATABASE_URL` | SQLAlchemy URL; defaults to SQLite at `instance/plant_disease.db` |
| `MODEL_PATH` | Optional compatible model; defaults to frozen Model V2 |
| `UPLOAD_FOLDER` | Defaults to `static/uploads` |
| `PREDICTION_CONFIDENCE_THRESHOLD` | Defaults to `60` |
| `WEATHER_API_KEY` | Reserved; no live weather provider is implemented yet |

Example PostgreSQL configuration:

```env
FLASK_ENV=production
SECRET_KEY=replace-with-a-long-random-secret
DATABASE_URL=postgresql+psycopg://username:password@localhost:5432/agridiagnose
```

Production mode fails clearly when `SECRET_KEY` or `DATABASE_URL` is missing.

## Database migrations

For a fresh database or after pulling new migrations:

```bash
flask --app run.py db upgrade
flask --app run.py db current
```

For a future schema change:

```bash
flask --app run.py db migrate -m "Describe the schema change"
# Review the generated migration before applying it.
flask --app run.py db upgrade
```

Do not delete an existing database automatically. If a SQLite database predates
Alembic, back it up and verify that its schema matches the intended migration
revision before using `db stamp`.

## REST API

### `POST /api/v1/predict`

The endpoint accepts one multipart image, performs the same validated inference
as the web application, removes its temporary file, and returns JSON. It is
currently public, stateless, and CSRF-exempt; add authentication and rate
limiting before exposing it as a public production service.

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
      "confidence": 94.72,
      "status": "diseased",
      "is_background": false,
      "uncertain": false
    },
    {
      "class_index": 30,
      "disease": "Tomato Early blight",
      "confidence": 3.81
    },
    {
      "class_index": 32,
      "disease": "Tomato Leaf Mold",
      "confidence": 1.12
    }
  ],
  "disease_info": {
    "plant_name": "Tomato",
    "disease_name": "Tomato : Late Blight",
    "description": "...",
    "symptoms": [],
    "causes": [],
    "treatment": ["..."],
    "prevention": [],
    "reference_image_url": "...",
    "disclaimer": "General guidance from the referenced dataset; consult local agricultural expertise."
  }
}
```

Invalid or corrupted images return a friendly `400` response. Uploads larger
than 5 MB are rejected by the application.

## Validation and CI

Install development dependencies and run the local quality gate:

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
pytest
flask --app run.py db current
flask --app run.py db check
git diff --check
```

GitHub Actions runs dependency validation, the full pytest suite, and Alembic
upgrade/current/check commands on Python 3.11. The Model V2 integration suite
also verifies the committed artifact checksum, `compile=False` loading,
224×224×3 input, 39-value finite synthetic output, preprocessing, threshold
semantics, and custom model overrides without using any TEST images.

## Documentation

- [Model V2 final evaluation and provenance](docs/model-v2-final-evaluation.md)
- [Frozen Model V2 baseline policy](training/config/model-v2-training-policy.json)
- [Frozen Experiment B augmentation policy](training/config/model-v2-experiment-b-policy.json)
- [Experiment A Kaggle workflow](docs/kaggle-model-v2-experiment-a.md)
- [Experiment B Kaggle workflow](docs/kaggle-model-v2-experiment-b.md)
- [Dataset V2 39-class composition](docs/dataset-v2-39class-composition.md)
- [Dataset V2 group-aware split](docs/dataset-v2-group-aware-split.md)
- [Conservative PlantDoc semantic mapping](evaluation/mappings/plantdoc.json)
- [Frozen machine-readable selection metadata](training/config/model-v2-final-selection.json)

## Limitations

- External PlantDoc performance shows significant domain shift.
- The external benchmark covers only 12 of 39 deployed classes.
- The background/no-leaf class was not evaluated by the PlantDoc subset.
- Confidence is not a guarantee of correctness or calibration in every domain.
- Disease guidance is general and has not replaced local agronomic review.
- The REST API does not yet include authentication, quotas, or rate limiting.
- A live weather provider and production hosting configuration are not included.
- Model V2 is frozen and must not be tuned using its final TEST observations.

## Roadmap

- Design a separately pre-declared Model V3 with more field-oriented training
  data, diverse cameras, backgrounds, lighting, and uncertainty handling.
- Preserve a new untouched external benchmark for final Model V3 evaluation.
- Implement and configure a resilient real weather provider.
- Add reviewed crop-specific agronomic metadata and stronger source citations.
- Add API authentication, rate limiting, observability, and a production WSGI
  deployment profile.

## Responsible use

AgriDiagnose AI is a software-engineering and machine-learning portfolio
project. It should support observation and triage, not replace laboratory
testing, agricultural extension services, or professional diagnosis.
