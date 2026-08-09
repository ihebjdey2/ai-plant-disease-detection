from datetime import datetime, timedelta

from app.extensions import db
from app.models.prediction import Prediction


def test_history_shows_only_current_users_predictions(
    authenticated_client, user, second_user, prediction_factory
):
    prediction_factory(user, disease="Apple healthy", status="healthy")
    prediction_factory(second_user, disease="Tomato mosaic virus", status="diseased")

    response = authenticated_client.get("/history")

    assert response.status_code == 200
    assert b"Apple healthy" in response.data
    assert b"Tomato mosaic virus" not in response.data


def test_prediction_detail_is_owner_scoped(
    authenticated_client, user, second_user, prediction_factory
):
    own = prediction_factory(user, disease="Apple healthy", status="healthy")
    other = prediction_factory(second_user, disease="Tomato Late blight")

    assert authenticated_client.get(f"/history/{own.id}").status_code == 200
    assert authenticated_client.get(f"/history/{other.id}").status_code == 404
    assert authenticated_client.get("/history/999999").status_code == 404


def test_delete_removes_selected_prediction_and_its_file(
    authenticated_client, app, user, second_user, prediction_factory
):
    selected = prediction_factory(user, disease="Apple healthy", status="healthy", create_image=True)
    remaining = prediction_factory(user, disease="Tomato Late blight")
    other = prediction_factory(second_user, disease="Potato Late blight")
    selected_path = app.config["UPLOAD_FOLDER"] / selected.image_path
    assert selected_path.exists()

    response = authenticated_client.post(f"/history/{selected.id}/delete")

    assert response.status_code == 302
    assert not selected_path.exists()
    with app.app_context():
        assert db.session.get(Prediction, selected.id) is None
        assert db.session.get(Prediction, remaining.id) is not None
        assert db.session.get(Prediction, other.id) is not None


def test_user_cannot_delete_another_users_prediction(
    authenticated_client, app, second_user, prediction_factory
):
    other = prediction_factory(second_user, create_image=True)
    other_path = app.config["UPLOAD_FOLDER"] / other.image_path

    response = authenticated_client.post(f"/history/{other.id}/delete")

    assert response.status_code == 404
    assert other_path.exists()
    with app.app_context():
        assert db.session.get(Prediction, other.id) is not None


def test_clear_history_only_removes_current_users_records_and_files(
    authenticated_client, app, user, second_user, prediction_factory
):
    own_one = prediction_factory(user, disease="Apple healthy", status="healthy", create_image=True)
    own_two = prediction_factory(user, disease="Tomato Late blight", create_image=True)
    other = prediction_factory(second_user, disease="Potato Late blight", create_image=True)
    own_paths = [app.config["UPLOAD_FOLDER"] / item.image_path for item in (own_one, own_two)]
    other_path = app.config["UPLOAD_FOLDER"] / other.image_path

    response = authenticated_client.post("/clear_history")

    assert response.status_code == 302
    assert all(not path.exists() for path in own_paths)
    assert other_path.exists()
    with app.app_context():
        assert Prediction.query.filter_by(user_id=user.id).count() == 0
        assert Prediction.query.filter_by(user_id=second_user.id).count() == 1


def test_history_pagination_is_ordered_and_owner_scoped(
    authenticated_client, user, second_user, prediction_factory
):
    start = datetime(2026, 1, 1, 8, 0)
    predictions = []
    for index in range(12):
        predictions.append(
            prediction_factory(
                user,
                disease=f"Disease-{index:03d}",
                created_at=start + timedelta(minutes=index),
            )
        )
    prediction_factory(
        second_user,
        disease="Other-user-private-disease",
        created_at=start + timedelta(days=1),
    )

    page_one = authenticated_client.get("/history?page=1")
    page_two = authenticated_client.get("/history?page=2")

    assert page_one.status_code == 200
    assert page_two.status_code == 200
    page_one_text = page_one.get_data(as_text=True)
    page_two_text = page_two.get_data(as_text=True)
    assert page_one_text.index("Disease-011") < page_one_text.index("Disease-010")
    assert "Disease-002" in page_one_text
    assert "Disease-001" not in page_one_text
    assert page_two_text.index("Disease-001") < page_two_text.index("Disease-000")
    assert "Disease-002" not in page_two_text
    assert "Other-user-private-disease" not in page_one_text
    assert "Other-user-private-disease" not in page_two_text
