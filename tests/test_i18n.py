from datetime import datetime

from app.i18n import format_datetime, translate_disease, translate_status


def test_language_switch_persists_and_enables_arabic_rtl(client):
    response = client.get("/language/ar?next=/login")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/login")
    page = client.get("/login").get_data(as_text=True)
    assert '<html lang="ar" dir="rtl">' in page
    assert "مرحبًا بعودتك" in page
    assert "تسجيل الدخول" in page


def test_french_interface_translates_dashboard_and_disease(
    authenticated_client, user, prediction_factory
):
    prediction_factory(user, disease="Tomato Late blight", status="diseased")
    authenticated_client.get("/language/fr?next=/")

    page = authenticated_client.get("/").get_data(as_text=True)
    assert '<html lang="fr" dir="ltr">' in page
    assert "Tableau de bord de la santé des cultures" in page
    assert "Plantes prises en charge" in page
    assert "Pommier" in page
    assert "Tomate" in page
    assert "38 classes de santé végétale" in page
    assert "Tomate — mildiou" in page
    assert "Malade" in page

    authenticated_client.get("/language/ar?next=/")
    arabic_page = authenticated_client.get("/").get_data(as_text=True)
    assert "النباتات التي يدعمها النموذج" in arabic_page
    assert "التفاح" in arabic_page
    assert "الطماطم" in arabic_page


def test_accept_language_is_used_when_no_preference_is_saved(client):
    response = client.get("/login", headers={"Accept-Language": "fr-FR,fr;q=0.9"})

    assert '<html lang="fr" dir="ltr">' in response.get_data(as_text=True)


def test_language_redirect_rejects_external_destination(client):
    response = client.get("/language/fr?next=//example.com")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")


def test_localized_helpers_cover_status_disease_and_date(app):
    with app.test_request_context("/", headers={"Accept-Language": "ar"}):
        assert translate_status("healthy") == "سليمة"
        assert translate_disease("Potato Late blight") == "البطاطا — اللفحة المتأخرة"
        assert format_datetime(datetime(2026, 8, 13, 14, 5)) == "13 أغسطس 2026, 14:05"


def test_validation_message_is_translated_in_french(client):
    client.get("/language/fr?next=/register")
    response = client.post(
        "/register",
        data={"name": "Test", "email": "test@example.com", "password": "court"},
        follow_redirects=True,
    )

    assert "mot de passe d’au moins 8 caractères" in response.get_data(as_text=True)


def test_modern_dashboard_keeps_accessible_interactions(authenticated_client):
    page = authenticated_client.get("/").get_data(as_text=True)
    css = authenticated_client.get("/static/css/style.css").get_data(as_text=True)

    assert 'class="skip-link" href="#main-content"' in page
    assert 'id="imageInput"' in page
    assert 'capture="environment"' in page
    assert 'aria-describedby="upload-help upload-status"' in page
    assert 'id="scanForm"' in page
    assert '<nav class="language-switcher"' in page
    assert 'onchange=' not in page
    assert "<bdi>39</bdi> model outputs including one no-leaf safety class" in page
    assert "prefers-reduced-motion: reduce" in css
    assert ":focus-visible" in css


def test_history_and_result_add_safe_human_centered_controls(
    authenticated_client, user, prediction_factory
):
    item = prediction_factory(user, disease="Tomato Late blight", status="diseased")

    history = authenticated_client.get("/history").get_data(as_text=True)
    detail = authenticated_client.get(f"/history/{item.id}").get_data(as_text=True)

    assert 'data-history-filter="all"' in history
    assert "This action cannot be undone." in history
    assert 'loading="lazy" decoding="async"' in history
    assert "Other model matches" in detail
    assert "not a guarantee of correctness" in detail
    assert 'class="confidence-ring"' in detail
