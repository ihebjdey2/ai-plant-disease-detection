from io import BytesIO

import pytest


def api_result(status, disease, confidence, class_index):
    primary = {
        "class_index": class_index,
        "disease": disease,
        "confidence": confidence,
        "status": status,
        "is_background": status == "no_leaf",
        "uncertain": status == "uncertain",
    }
    return {"prediction": primary, "top_predictions": [primary.copy()]}


@pytest.mark.parametrize(
    ("status", "disease", "confidence", "class_index"),
    [
        ("healthy", "Apple healthy", 91.0, 3),
        ("diseased", "Tomato Late blight", 94.0, 31),
        ("uncertain", "Tomato Late blight", 42.0, 31),
        ("no_leaf", "Background without leaves", 99.0, 4),
    ],
)
def test_predict_api_returns_structured_status_and_cleans_temp_file(
    client,
    app,
    image_factory,
    mocker,
    status,
    disease,
    confidence,
    class_index,
):
    mocker.patch(
        "app.routes.api.predict",
        return_value=api_result(status, disease, confidence, class_index),
    )

    response = client.post(
        "/api/v1/predict",
        data={"image": (image_factory("JPEG"), "leaf.jpg")},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["prediction"]["class_index"] == class_index
    assert payload["prediction"]["disease"] == disease
    assert payload["prediction"]["confidence"] == confidence
    assert payload["prediction"]["status"] == status
    assert payload["top_predictions"]
    assert list(app.config["UPLOAD_FOLDER"].iterdir()) == []


def test_predict_api_rejects_missing_image(client):
    response = client.post("/api/v1/predict")

    assert response.status_code == 400
    assert response.get_json() == {"success": False, "error": "image is required"}


def test_predict_api_rejects_corrupt_image_without_leaving_files(client, app):
    response = client.post(
        "/api/v1/predict",
        data={"image": (BytesIO(b"corrupted image content"), "leaf.jpg")},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["success"] is False
    assert "not a valid image" in payload["error"]
    assert "Traceback" not in payload["error"]
    assert list(app.config["UPLOAD_FOLDER"].iterdir()) == []
