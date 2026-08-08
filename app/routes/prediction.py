from pathlib import Path
from uuid import uuid4
from flask import Blueprint, current_app, flash, redirect, request, url_for
from flask_login import current_user, login_required
from app.extensions import db
from app.models.prediction import Prediction
from app.services.prediction_service import PredictionError, predict
prediction_bp=Blueprint("prediction",__name__)
@prediction_bp.post("/scan")
@login_required
def create():
    file=request.files.get("image")
    if not file or not file.filename: flash("Choose an image to analyze.","error"); return redirect(url_for("dashboard.index"))
    ext=file.filename.rsplit(".",1)[-1].lower() if "." in file.filename else ""
    if ext not in current_app.config["ALLOWED_IMAGE_EXTENSIONS"]: flash("Supported formats: JPG, PNG, WEBP.","error"); return redirect(url_for("dashboard.index"))
    filename=f"{uuid4().hex}.{ext}"; path=Path(current_app.config["UPLOAD_FOLDER"])/filename; file.save(path)
    try: result=predict(path)
    except PredictionError as exc: path.unlink(missing_ok=True); flash(str(exc),"error"); return redirect(url_for("dashboard.index"))
    item=Prediction(user_id=current_user.id,disease=result["prediction"]["disease"],confidence=result["prediction"]["confidence"],image_path=filename,top_predictions=result["top_predictions"]); db.session.add(item); db.session.commit()
    return redirect(url_for("history.detail",prediction_id=item.id))
