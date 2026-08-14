from flask import Blueprint, render_template
from flask_login import current_user, login_required
from sqlalchemy import func
from app.extensions import db
from app.models.prediction import Prediction
from app.taxonomy import supported_crops
dashboard_bp=Blueprint("dashboard",__name__)
@dashboard_bp.get("/")
@login_required
def index():
    q=Prediction.query.filter_by(user_id=current_user.id)
    total=q.count()
    counts=dict(q.with_entities(Prediction.status,func.count(Prediction.id)).group_by(Prediction.status).all())
    frequent=q.filter(Prediction.status=="diseased").with_entities(Prediction.disease,func.count(Prediction.id).label("count")).group_by(Prediction.disease).order_by(func.count(Prediction.id).desc()).limit(3).all()
    crops=supported_crops()
    return render_template("index.html", total_scans=total, healthy_scans=counts.get("healthy",0), diseased_scans=counts.get("diseased",0), uncertain_scans=counts.get("uncertain",0), no_leaf_scans=counts.get("no_leaf",0), average_confidence=q.with_entities(func.avg(Prediction.confidence)).scalar() or 0, recent_predictions=q.order_by(Prediction.created_at.desc()).limit(5).all(), frequent_diseases=frequent, supported_crops=crops, plant_class_count=sum(crop["class_count"] for crop in crops))
