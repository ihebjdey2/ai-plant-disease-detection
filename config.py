"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "agri-diagnose-v2-exp-a.keras"
DEFAULT_MODEL_SELECTION_PATH = (
    BASE_DIR / "training" / "config" / "model-v2-final-selection.json"
)
load_dotenv(BASE_DIR / ".env")
MODEL_PATH_OVERRIDE = os.getenv("MODEL_PATH")


class Config:
    ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = ENV == "development"
    SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_urlsafe(32)
    MODEL_PATH = (
        Path(MODEL_PATH_OVERRIDE) if MODEL_PATH_OVERRIDE else DEFAULT_MODEL_PATH
    )
    MODEL_PATH_IS_DEFAULT = not bool(MODEL_PATH_OVERRIDE)
    MODEL_SELECTION_PATH = DEFAULT_MODEL_SELECTION_PATH
    UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", BASE_DIR / "static" / "uploads"))
    HISTORY_FILE = Path(os.getenv("HISTORY_FILE", BASE_DIR / "history.json"))
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024
    ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    PREDICTION_CONFIDENCE_THRESHOLD = float(
        os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "60")
    )
    DATABASE_URL = os.getenv("DATABASE_URL")
    SQLALCHEMY_DATABASE_URI = DATABASE_URL or f"sqlite:///{BASE_DIR / 'instance' / 'plant_disease.db'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = 60 * 60 * 8


class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True

    @classmethod
    def validate(cls) -> None:
        missing = [name for name in ("SECRET_KEY", "DATABASE_URL") if not os.getenv(name)]
        if missing:
            raise RuntimeError(f"Missing required production environment variables: {', '.join(missing)}")


class TestingConfig(Config):
    """Safe defaults for automated tests; fixtures may override temporary paths."""

    ENV = "testing"
    TESTING = True
    DEBUG = False
    SECRET_KEY = "agridiagnose-test-secret"
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False
    WEATHER_API_KEY = None
