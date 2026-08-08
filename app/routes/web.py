"""Compatibility web routes retained while persistence is migrated in phase 2."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Blueprint, current_app, redirect, render_template, request, session
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

from config import Config
from utils.disease_api import get_disease_info
from utils.weather import get_weather


web_bp = Blueprint("web", __name__)
logger = logging.getLogger(__name__)

# Kept unchanged in phase 1. Phase 3 will replace this with the validated service.
MODEL_PATH = Config.MODEL_PATH
model = load_model(MODEL_PATH, compile=False)
logger.info("Plant disease model loaded")

class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

# Transitional only; replaced by the User database model in phase 2.
users: dict[str, dict[str, str]] = {}


def _history_file() -> Path:
    return Path(current_app.config["HISTORY_FILE"])


def load_history() -> list[dict]:
    history_file = _history_file()
    if not history_file.exists():
        history_file.write_text("[]", encoding="utf-8")
    return json.loads(history_file.read_text(encoding="utf-8"))


def save_history(data: list[dict]) -> None:
    _history_file().write_text(json.dumps(data, indent=4), encoding="utf-8")


@web_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        user = users.get(email)
        if user and user["password"] == request.form["password"]:
            session["user"] = email
            return redirect("/")
        return render_template("login.html", error="Invalid Email or Password")
    return render_template("login.html")


@web_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        users[request.form["email"]] = {"name": request.form["name"], "password": request.form["password"]}
        return redirect("/login")
    return render_template("register.html")


@web_bp.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")


@web_bp.route("/", methods=["GET", "POST"])
def home():
    if "user" not in session:
        return redirect("/login")
    result = None
    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filename = f"{datetime.now():%Y%m%d%H%M%S}_{secure_filename(file.filename)}"
            filepath = Path(current_app.config["UPLOAD_FOLDER"]) / filename
            file.save(filepath)
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = np.expand_dims(image.img_to_array(img), axis=0)
            prediction = model.predict(preprocess_input(img_array), verbose=0)
            predicted_index = int(np.argmax(prediction))
            disease = class_names[predicted_index].replace("___", " ").replace("_", " ")
            try:
                disease_info = get_disease_info(disease)
            except (KeyError, TypeError, ValueError):
                logger.exception("Could not retrieve disease information")
                disease_info = "No information available."
            try:
                weather = get_weather()
            except TypeError:
                weather = None
            result = {
                "disease": disease,
                "confidence": round(float(np.max(prediction)) * 100, 2),
                "info": disease_info,
                "image": f"static/uploads/{filename}",
                "weather": weather,
                "date": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            }
            history = load_history()
            history.insert(0, result)
            save_history(history)
    return render_template("index.html", result=result)


@web_bp.route("/history")
def history():
    if "user" not in session:
        return redirect("/login")
    return render_template("history.html", history=load_history())


@web_bp.route("/clear_history", methods=["POST"])
def clear_history():
    if "user" not in session:
        return redirect("/login")
    for item in load_history():
        image_path = item.get("image")
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                logger.exception("Could not delete uploaded image: %s", image_path)
    save_history([])
    return redirect("/history")
