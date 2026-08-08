from pathlib import Path
from flask import Blueprint, current_app, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from app.extensions import db
from app.models.prediction import Prediction
history_bp=Blueprint("history",__name__)
@history_bp.get("/history")
@login_required
def index():
    page=request.args.get("page", 1, type=int)
    history=Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).paginate(page=page, per_page=10, error_out=False)
    return render_template("history.html",history=history)
@history_bp.get("/history/<int:prediction_id>")
@login_required
def detail(prediction_id):
    item=Prediction.query.filter_by(id=prediction_id,user_id=current_user.id).first_or_404()
    return render_template("detail.html", item=item)
@history_bp.post("/history/<int:prediction_id>/delete")
@login_required
def delete(prediction_id):
    page=request.form.get("page",1,type=int); item=Prediction.query.filter_by(id=prediction_id,user_id=current_user.id).first_or_404(); (Path(current_app.config["UPLOAD_FOLDER"])/item.image_path).unlink(missing_ok=True); db.session.delete(item); db.session.commit(); return redirect(url_for("history.index",page=page))
@history_bp.post("/clear_history")
@login_required
def clear():
    for item in Prediction.query.filter_by(user_id=current_user.id): (Path(current_app.config["UPLOAD_FOLDER"])/item.image_path).unlink(missing_ok=True); db.session.delete(item)
    db.session.commit(); return redirect(url_for("history.index"))
