from app.extensions import db
from app.models.user import User


def test_login_and_register_pages_are_available(client):
    assert client.get("/login").status_code == 200
    assert client.get("/register").status_code == 200


def test_valid_registration_hashes_password(client, app):
    response = client.post(
        "/register",
        data={"name": "New User", "email": "NEW@Example.com", "password": "secure123"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")
    with app.app_context():
        account = User.query.filter_by(email="new@example.com").one()
        assert account.password_hash != "secure123"
        assert account.check_password("secure123")


def test_duplicate_email_is_rejected(client, user, app):
    response = client.post(
        "/register",
        data={"name": "Duplicate", "email": user.email.upper(), "password": "secure123"},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"That email is already registered." in response.data
    with app.app_context():
        assert db.session.query(User).count() == 1


def test_short_password_is_rejected(client, app):
    response = client.post(
        "/register",
        data={"name": "New User", "email": "new@example.com", "password": "short"},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"password of at least 8 characters" in response.data
    with app.app_context():
        assert db.session.query(User).count() == 0


def test_valid_login_succeeds(client, user):
    response = client.post(
        "/login", data={"email": user.email, "password": "strong-pass"}
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")


def test_invalid_password_is_rejected(client, user):
    response = client.post(
        "/login",
        data={"email": user.email, "password": "wrong-password"},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Invalid email or password." in response.data


def test_unknown_email_is_rejected(client):
    response = client.post(
        "/login",
        data={"email": "missing@example.com", "password": "strong-pass"},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Invalid email or password." in response.data


def test_logout_removes_access_to_protected_page(authenticated_client):
    assert authenticated_client.get("/").status_code == 200

    response = authenticated_client.post("/logout")

    assert response.status_code == 302
    protected = authenticated_client.get("/")
    assert protected.status_code == 302
    assert "/login" in protected.headers["Location"]


def test_protected_routes_redirect_anonymous_users(client):
    requests = [
        client.get("/"),
        client.get("/history"),
        client.get("/history/1"),
        client.post("/scan"),
    ]

    for response in requests:
        assert response.status_code == 302
        assert "/login" in response.headers["Location"]
