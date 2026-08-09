from io import BytesIO

import pytest

from app.models.prediction import Prediction
from app.services.prediction_service import PredictionError


def prediction_result(status, disease, confidence, class_index):
    primary = {
        "class_index": class_index,
        "disease": disease,
        "confidence": confidence,
        "is_background": status == "no_leaf",
        "uncertain": status == "uncertain",
        "status": status,
    }
    return {"prediction": primary, "top_predictions": [primary.copy()]}


@pytest.mark.parametrize(
    ("status", "disease", "confidence", "class_index"),
    [
        ("healthy", "Bell pepper healthy", 92.0, 20),
        ("diseased", "Tomato Late blight", 95.0, 31),
        ("uncertain", "Tomato Late blight", 38.0, 31),
        ("no_leaf", "Background without leaves", 98.0, 4),
    ],
)
def test_scan_persists_each_service_status(
    authenticated_client,
    app,
    image_factory,
    mocker,
    status,
    disease,
    confidence,
    class_index,
):
    result = prediction_result(status, disease, confidence, class_index)
    mocker.patch("app.routes.prediction.predict", return_value=result)

    response = authenticated_client.post(
        "/scan", data={"image": (image_factory("JPEG"), "leaf.jpg")}
    )

    assert response.status_code == 302
    with app.app_context():
        prediction = Prediction.query.one()
        assert prediction.status == status
        assert prediction.disease == disease
        assert prediction.confidence == confidence
        assert prediction.top_predictions == result["top_predictions"]


def test_scan_does_not_recompute_status_from_disease(
    authenticated_client, app, image_factory, mocker
):
    result = prediction_result("uncertain", "Bell pepper healthy", 99.0, 20)
    mocker.patch("app.routes.prediction.predict", return_value=result)

    authenticated_client.post(
        "/scan", data={"image": (image_factory("JPEG"), "leaf.jpg")}
    )

    with app.app_context():
        assert Prediction.query.one().status == "uncertain"


@pytest.mark.parametrize(
    ("image_format", "filename"), [("JPEG", "leaf.jpg"), ("PNG", "leaf.png")]
)
def test_valid_jpg_and_png_uploads_are_accepted(
    authenticated_client, app, image_factory, mocker, image_format, filename
):
    mocker.patch(
        "app.routes.prediction.predict",
        return_value=prediction_result("healthy", "Apple healthy", 90.0, 3),
    )

    response = authenticated_client.post(
        "/scan", data={"image": (image_factory(image_format), filename)}
    )

    assert response.status_code == 302
    with app.app_context():
        assert Prediction.query.count() == 1


def test_missing_image_is_rejected(authenticated_client, app):
    response = authenticated_client.post("/scan", follow_redirects=True)

    assert response.status_code == 200
    assert b"Choose an image to analyze." in response.data
    with app.app_context():
        assert Prediction.query.count() == 0


def test_unsupported_extension_is_rejected(
    authenticated_client, app, image_factory, mocker
):
    predict_mock = mocker.patch("app.routes.prediction.predict")

    response = authenticated_client.post(
        "/scan",
        data={"image": (image_factory("PNG"), "leaf.gif")},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Supported formats: JPG, PNG, WEBP." in response.data
    predict_mock.assert_not_called()
    with app.app_context():
        assert Prediction.query.count() == 0


def test_corrupted_image_is_handled_and_removed(authenticated_client, app):
    response = authenticated_client.post(
        "/scan",
        data={"image": (BytesIO(b"not a real image"), "leaf.jpg")},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"not a valid image" in response.data
    assert list(app.config["UPLOAD_FOLDER"].iterdir()) == []
    with app.app_context():
        assert Prediction.query.count() == 0


def test_prediction_error_returns_safe_message_and_removes_upload(
    authenticated_client, app, image_factory, mocker
):
    mocker.patch(
        "app.routes.prediction.predict",
        side_effect=PredictionError("The image could not be analyzed."),
    )

    response = authenticated_client.post(
        "/scan",
        data={"image": (image_factory("JPEG"), "leaf.jpg")},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"The image could not be analyzed." in response.data
    assert b"Traceback" not in response.data
    assert list(app.config["UPLOAD_FOLDER"].iterdir()) == []


def test_oversized_upload_returns_413(authenticated_client, app):
    oversized = BytesIO(b"x" * (app.config["MAX_CONTENT_LENGTH"] + 1024))

    response = authenticated_client.post(
        "/scan", data={"image": (oversized, "large.jpg")}
    )

    assert response.status_code == 413
    assert b"Images must be 5 MB or smaller." in response.data
