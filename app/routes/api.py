from pathlib import Path
from tempfile import NamedTemporaryFile
from flask import Blueprint, current_app, jsonify, request
from app.extensions import csrf
from app.services.disease_service import get_disease_info
from app.services.prediction_service import PredictionError, predict
api_bp=Blueprint("api",__name__,url_prefix="/api/v1")
@api_bp.post("/predict")
@csrf.exempt
def predict_api():
    file=request.files.get("image")
    if not file: return jsonify(success=False,error="image is required"),400
    suffix=Path(file.filename or "image.jpg").suffix
    with NamedTemporaryFile(suffix=suffix,delete=False,dir=current_app.config["UPLOAD_FOLDER"]) as tmp: file.save(tmp.name); path=Path(tmp.name)
    try:
        result=predict(path)
        return jsonify(success=True, prediction=result["prediction"], top_predictions=result["top_predictions"], disease_info=get_disease_info(result["prediction"]["disease"]))
    except PredictionError as exc: return jsonify(success=False,error=str(exc)),400
    finally: path.unlink(missing_ok=True)
