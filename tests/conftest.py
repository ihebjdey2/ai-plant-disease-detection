from __future__ import annotations

from datetime import datetime
from io import BytesIO
from uuid import uuid4

import pytest
from PIL import Image

from app import create_app
from app.extensions import db
from app.models.prediction import Prediction
from app.models.user import User
from config import TestingConfig


@pytest.fixture
def app(tmp_path):
    class TemporaryTestingConfig(TestingConfig):
        UPLOAD_FOLDER = tmp_path / "uploads"

    application = create_app(TemporaryTestingConfig)
    assert application.config["SQLALCHEMY_DATABASE_URI"] == "sqlite:///:memory:"
    assert application.config["UPLOAD_FOLDER"].parent == tmp_path

    with application.app_context():
        db.create_all()

    yield application

    with application.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def db_session(app):
    with app.app_context():
        yield db.session


@pytest.fixture
def user(app):
    with app.app_context():
        account = User(name="Test User", email="user@example.com")
        account.set_password("strong-pass")
        db.session.add(account)
        db.session.commit()
        db.session.refresh(account)
        return account


@pytest.fixture
def second_user(app):
    with app.app_context():
        account = User(name="Other User", email="other@example.com")
        account.set_password("other-pass")
        db.session.add(account)
        db.session.commit()
        db.session.refresh(account)
        return account


@pytest.fixture
def authenticated_client(client, user):
    response = client.post(
        "/login",
        data={"email": user.email, "password": "strong-pass"},
    )
    assert response.status_code == 302
    return client


@pytest.fixture
def image_factory():
    def make_image(image_format="JPEG"):
        stream = BytesIO()
        Image.new("RGB", (8, 8), color=(40, 150, 70)).save(stream, format=image_format)
        stream.seek(0)
        return stream

    return make_image


@pytest.fixture
def prediction_factory(app):
    def create_prediction(
        owner,
        *,
        disease="Tomato Late blight",
        confidence=90.0,
        status="diseased",
        created_at: datetime | None = None,
        create_image=False,
    ):
        filename = f"prediction-{owner.id}-{uuid4().hex}.jpg"
        if create_image:
            upload_path = app.config["UPLOAD_FOLDER"] / filename
            upload_path.write_bytes(b"test-image")
        with app.app_context():
            prediction = Prediction(
                user_id=owner.id,
                disease=disease,
                confidence=confidence,
                image_path=filename,
                top_predictions=[
                    {"class_index": 31, "disease": disease, "confidence": confidence}
                ],
                status=status,
            )
            if created_at is not None:
                prediction.created_at = created_at
            db.session.add(prediction)
            db.session.commit()
            db.session.refresh(prediction)
            return prediction

    return create_prediction
