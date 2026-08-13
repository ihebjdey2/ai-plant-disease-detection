# AgriDiagnose AI — Plant Disease Detection

[![CI](https://github.com/ihebjdey2/ai-plant-disease-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/ihebjdey2/ai-plant-disease-detection/actions/workflows/ci.yml)

Flask application for AI-assisted plant leaf analysis. It uses the included MobileNetV2-based TensorFlow model to provide decision-support predictions, a secure account area, scan history, and a REST endpoint for future mobile clients.

> Predictions are not confirmed agricultural diagnoses. Validate results with local agronomy guidance before treatment decisions.

## Features

- Secure registration, login, logout, hashed passwords, and CSRF protection
- Per-user prediction history stored through SQLAlchemy
- PostgreSQL configuration for production; SQLite fallback for local development
- Validated JPG, JPEG, PNG, and WEBP uploads up to 5 MB
- TensorFlow model loaded once and shared by web/API prediction flows
- Confidence threshold and top-three output
- Versioned API: `POST /api/v1/predict`

## Architecture

```text
app/
  routes/       HTTP blueprints
  models/       SQLAlchemy models
  services/     prediction, disease and weather logic
  extensions.py Flask extensions
config.py       environment-based configuration
run.py          application entry point
```

## Setup

Python 3.11 and TensorFlow 2.15 are required. On Windows, create the virtual environment in a short path if TensorFlow hits the Windows path-length limit.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python run.py
```

Open `http://127.0.0.1:5000`.

## Configuration

Set values in `.env`; never commit it.

| Variable | Purpose |
| --- | --- |
| `SECRET_KEY` | Required long random value in production |
| `DATABASE_URL` | PostgreSQL SQLAlchemy URL, e.g. `postgresql+psycopg://user:password@host:5432/agridiagnose` |
| `MODEL_PATH` | Optional custom model path; defaults to frozen Model V2 |
| `WEATHER_API_KEY` | Reserved for a configured weather provider |
| `PREDICTION_CONFIDENCE_THRESHOLD` | Default: `60` |

## REST API

```bash
curl -X POST http://127.0.0.1:5000/api/v1/predict -F "image=@leaf.jpg"
```

The response contains `prediction`, `top_predictions`, and cautious `disease_info` guidance.

## Model class mapping

The model has 39 outputs. The complete class order includes `Background_without_leaves` at index 4; this mapping was recovered from the companion PlantVillage project’s documented `idx_to_classes` dictionary and matches the 39-output shape. A background prediction is presented as “No leaf detected” and does not receive disease guidance.

## Model V2 final evaluation

The application now uses the exact frozen `models/agri-diagnose-v2-exp-a.keras`
artifact by default. The legacy `plant_disease_model.h5` remains in the
repository for explicit rollback. A custom `MODEL_PATH` override is still
supported.

Model V2 was selected strictly on VALIDATION. On the held-out 7,344-image,
39-class Dataset V2 INTERNAL TEST it reached 95.44% accuracy and 95.92%
Macro-F1. On the conservative 99-image / 12-class external PlantDoc TEST
overlap subset it reached 41.41% accuracy and 45.79% Macro-F1.

The external result demonstrates substantial domain shift; the 95.44% internal
result is not a field-accuracy claim. See the
[frozen Model V2 evaluation and provenance](docs/model-v2-final-evaluation.md).

## Future improvements

- Develop a separately pre-declared, field-oriented Model V3 and preserve a new
  untouched external benchmark for its final evaluation.
- Configure a real weather provider.
- Add database migrations and automated integration tests.
- Add crop-specific guidance reviewed by an agricultural specialist.
