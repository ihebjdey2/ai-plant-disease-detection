from flask import Blueprint, render_template
from flask_login import current_user, login_required
from sqlalchemy import func
from app.extensions import db
from app.models.prediction import Prediction
dashboard_bp=Blueprint("dashboard",__name__)
@dashboard_bp.get("/")
@login_required
def index():
    q=Prediction.query.filter_by(user_id=current_user.id)
    total=q.count(); healthy=q.filter(Prediction.disease.ilike("%healthy%")).count()
    frequent=q.with_entities(Prediction.disease,func.count(Prediction.id).label("count")).group_by(Prediction.disease).order_by(func.count(Prediction.id).desc()).limit(3).all()
    return render_template("index.html", total_scans=total, healthy_scans=healthy, diseased_scans=total-healthy, average_confidence=q.with_entities(func.avg(Prediction.confidence)).scalar() or 0, recent_predictions=q.order_by(Prediction.created_at.desc()).limit(5).all(), frequent_diseases=frequent)
