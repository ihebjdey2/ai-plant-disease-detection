

<div align="center">
# AgriDiagnose AI
    
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

AgriDiagnose AI is a full-stack machine-learning application for plant leaf analysis.

It combines a frozen **MobileNetV2 classifier** with a Flask web application providing authentication, prediction history, dashboards, multilingual support, and a REST API.

The project also includes reproducible dataset construction, model evaluation, automated tests, database migrations, CI, and scientific documentation.

---

## Key Results

| Metric | Result |
|---|---:|
| Dataset V2 | **73,563 audited images** |
| Training | **58,857 images** |
| Validation | **7,362 images** |
| Internal Test | **7,344 images** |
| Classes | **39** |
| Crops | **14** |
| Internal Test Accuracy | **95.44%** |
| Internal Macro-F1 | **95.92%** |
| External PlantDoc Accuracy | **41.41%** |

The gap between internal and external evaluation highlights a significant **domain shift** and shows why the internal score should not be interpreted as field accuracy. 

---

## Model

```text
Input Image
    │
    ▼
224 × 224 RGB
    │
    ▼
MobileNetV2
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
Dropout
    │
    ▼
Dense Softmax
    │
    ▼
39 Classes
```

Frozen model:

```text
models/agri-diagnose-v2-exp-a.keras
```

Key properties:

- MobileNetV2
- 39 output classes
- RGB `224 × 224`
- `/255.0` preprocessing
- 60% confidence threshold
- SHA-256 integrity verification

---

## Features

### Plant Analysis

- JPG, PNG and WEBP image upload
- Top-3 predictions
- Healthy, diseased, uncertain and no-leaf states
- Disease information, symptoms and guidance
- Image validation before inference

### User Experience

- Registration and login
- Prediction history
- User dashboard
- Prediction details
- Responsive interface

### Multilingual UI

- English
- French
- Arabic with RTL support

### Engineering

- Flask application factory
- Modular Blueprints
- REST API
- SQLAlchemy persistence
- PostgreSQL support
- Alembic migrations
- pytest
- GitHub Actions CI

---

## Architecture

```text
                Browser / REST API
                        │
                        ▼
                   Flask Routes
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
      Prediction Service      SQLAlchemy
              │                   │
              ▼                   ▼
       MobileNetV2 Model        Database
              │
              ▼
         39-Class Output
```

---

## Tech Stack

<p>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg" width="42" height="42" alt="Python" />
  &nbsp;&nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/tensorflow/tensorflow-original.svg" width="42" height="42" alt="TensorFlow" />
  &nbsp;&nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/flask/flask-original.svg" width="42" height="42" alt="Flask" />
  &nbsp;&nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/postgresql/postgresql-original.svg" width="42" height="42" alt="PostgreSQL" />
</p>

### AI & Computer Vision

- Python 3.11
- TensorFlow 2.15
- Keras
- MobileNetV2
- OpenCV
- NumPy
- pandas
- scikit-learn

### Backend

- Flask
- SQLAlchemy
- Flask-Login
- Flask-WTF
- Flask-Migrate
- REST API

### Data & Quality

- SQLite / PostgreSQL
- pytest
- GitHub Actions
- Alembic
- SHA-256 model verification

---

## Project Structure

```text
app/
├── data/
├── models/
├── routes/
└── services/

models/
training/
evaluation/
notebooks/
scripts/
docs/
tests/
migrations/

config.py
run.py
```

---

## Quick Start

### Requirements

- Python 3.11
- Git

### Clone

```bash
git clone https://github.com/ihebjdey2/ai-plant-disease-detection.git
cd ai-plant-disease-detection
```

### Environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### Install

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Database

```bash
flask --app run.py db upgrade
```

### Run

```bash
python run.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## REST API

Prediction endpoint:

```http
POST /api/v1/predict
```

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/v1/predict \
  -F "image=@leaf.jpg"
```

---

## Testing

```bash
python -m pip install -r requirements-dev.txt
pytest
```

GitHub Actions validates the project on Python 3.11.

---

## Documentation

Detailed dataset, training and evaluation documentation is available under:

```text
docs/
```

Including:

- Model V2 evaluation and provenance
- Dataset V2 composition
- Group-aware split methodology
- Experiment A/B workflows
- PlantDoc external evaluation

---
<div align="center">

### AgriDiagnose AI

**73,563 audited images · 39 classes · 14 crops · 3 languages**

Built with **Flask · TensorFlow · MobileNetV2 · PostgreSQL · GitHub Actions**

</div>
