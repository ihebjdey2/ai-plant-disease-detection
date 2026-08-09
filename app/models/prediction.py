from __future__ import annotations

from datetime import datetime

from app.extensions import db


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    disease = db.Column(db.String(255), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    top_predictions = db.Column(db.JSON, nullable=True)
    status = db.Column(db.String(20), nullable=False, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    user = db.relationship("User", back_populates="predictions")
