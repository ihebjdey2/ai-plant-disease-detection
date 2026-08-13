"""Flask application factory."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask, redirect, render_template, request, session, url_for
from config import Config
from app.extensions import csrf, db, login_manager, migrate


def create_app(config_object: type[Config] = Config) -> Flask:
    project_root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    app.config.from_object(config_object)
    if app.config["ENV"] == "production":
        ProductionConfig = __import__("config", fromlist=["ProductionConfig"]).ProductionConfig
        ProductionConfig.validate()

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    if not app.config["TESTING"]:
        _configure_logging(app, project_root / "instance")

    if not app.config.get("SECRET_KEY"):
        raise RuntimeError("SECRET_KEY must be configured.")
    if not app.config["TESTING"] and not __import__("os").getenv("SECRET_KEY"):
        app.logger.warning(
            "SECRET_KEY is not configured; a temporary development key was generated."
        )

    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    csrf.init_app(app)

    from app.models.user import User

    @login_manager.user_loader
    def load_user(user_id: str) -> User | None:
        return db.session.get(User, int(user_id))

    from app.routes.api import api_bp
    from app.routes.auth import auth_bp
    from app.routes.dashboard import dashboard_bp
    from app.routes.history import history_bp
    from app.routes.prediction import prediction_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(api_bp)

    from app.i18n import (
        RTL_LANGUAGES,
        SUPPORTED_LANGUAGES,
        format_datetime,
        get_locale,
        translate,
        translate_disease,
        translate_status,
    )

    @app.context_processor
    def inject_i18n():
        locale = get_locale()
        return {
            "t": translate,
            "current_language": locale,
            "language_direction": "rtl" if locale in RTL_LANGUAGES else "ltr",
            "supported_languages": SUPPORTED_LANGUAGES,
        }

    app.jinja_env.filters["status_name"] = translate_status
    app.jinja_env.filters["disease_name"] = translate_disease
    app.jinja_env.filters["local_datetime"] = format_datetime

    @app.get("/language/<language>")
    def set_language(language: str):
        if language in SUPPORTED_LANGUAGES:
            session["language"] = language
        destination = request.args.get("next", "")
        if not destination.startswith("/") or destination.startswith("//"):
            destination = url_for("dashboard.index")
        return redirect(destination)

    @app.errorhandler(403)
    def forbidden(_error): return render_template("error.html", code=403, message=translate("You do not have access to this resource.")), 403
    @app.errorhandler(404)
    def not_found(_error): return render_template("error.html", code=404, message=translate("The page was not found.")), 404
    @app.errorhandler(413)
    def too_large(_error): return render_template("error.html", code=413, message=translate("Images must be 5 MB or smaller.")), 413
    @app.errorhandler(500)
    def server_error(error): app.logger.exception("Unhandled server error: %s", error); return render_template("error.html", code=500, message=translate("Something went wrong. Please try again.")), 500
    return app


def _configure_logging(app: Flask, instance_dir: Path) -> None:
    instance_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        instance_dir / "application.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
