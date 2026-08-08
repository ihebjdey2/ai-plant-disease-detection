# Phasal AI — Plant Disease Detection

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
| `DATABASE_URL` | PostgreSQL SQLAlchemy URL, e.g. `postgresql+psycopg://user:password@host:5432/phasal` |
| `MODEL_PATH` | Path to `plant_disease_model.h5` |
| `WEATHER_API_KEY` | Reserved for a configured weather provider |
| `PREDICTION_CONFIDENCE_THRESHOLD` | Default: `60` |

## REST API

```bash
curl -X POST http://127.0.0.1:5000/api/v1/predict -F "image=@leaf.jpg"
```

The response contains `prediction`, `top_predictions`, and cautious `disease_info` guidance.

## Model limitation

The checked-in model has 39 output units, while this repository supplies only 38 class labels. The unmapped output is deliberately reported as `Unmapped model class`; it is not guessed. Obtain the original `train_generator.class_indices` mapping before deploying predictions as a reliable classification feature.

## Future improvements

- Add the verified 39-class label mapping and model evaluation metrics.
- Configure a real weather provider.
- Add database migrations and automated integration tests.
- Add crop-specific guidance reviewed by an agricultural specialist.
