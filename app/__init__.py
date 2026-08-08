"""Flask application factory."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask
from sqlalchemy import inspect, text

from config import Config
from app.extensions import csrf, db, login_manager


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
    _configure_logging(app, project_root / "instance")

    if not app.config.get("SECRET_KEY"):
        raise RuntimeError("SECRET_KEY must be configured.")
    if not __import__("os").getenv("SECRET_KEY"):
        app.logger.warning(
            "SECRET_KEY is not configured; a temporary development key was generated."
        )

    db.init_app(app)
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

    with app.app_context():
        db.create_all()
        # Small, backward-compatible development migration. Production databases
        # should use a proper migration tool before deployment.
        if app.config["SQLALCHEMY_DATABASE_URI"].startswith("sqlite"):
            columns = {column["name"] for column in inspect(db.engine).get_columns("prediction")}
            if "top_predictions" not in columns:
                db.session.execute(text("ALTER TABLE prediction ADD COLUMN top_predictions JSON"))
                db.session.commit()
    @app.errorhandler(403)
    def forbidden(_error): return __import__("flask").render_template("error.html", code=403, message="You do not have access to this resource."), 403
    @app.errorhandler(404)
    def not_found(_error): return __import__("flask").render_template("error.html", code=404, message="The page was not found."), 404
    @app.errorhandler(413)
    def too_large(_error): return __import__("flask").render_template("error.html", code=413, message="Images must be 5 MB or smaller."), 413
    @app.errorhandler(500)
    def server_error(error): app.logger.exception("Unhandled server error: %s", error); return __import__("flask").render_template("error.html", code=500, message="Something went wrong. Please try again."), 500
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
