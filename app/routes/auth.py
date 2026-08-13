from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_user, logout_user
from app.extensions import db
from app.models.user import User
from app.i18n import translate
auth_bp = Blueprint("auth", __name__)
@auth_bp.route("/login", methods=["GET","POST"])
def login():
    if current_user.is_authenticated: return redirect(url_for("dashboard.index"))
    if request.method == "POST":
        user = User.query.filter_by(email=request.form.get("email", "").strip().lower()).first()
        if user and user.check_password(request.form.get("password", "")):
            login_user(user); return redirect(url_for("dashboard.index"))
        flash(translate("Invalid email or password."), "error")
    return render_template("login.html")
@auth_bp.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        email=request.form.get("email", "").strip().lower(); password=request.form.get("password", ""); name=request.form.get("name", "").strip()
        if not name or not email or len(password)<8: flash(translate("Use a name, valid email, and password of at least 8 characters."),"error")
        elif User.query.filter_by(email=email).first(): flash(translate("That email is already registered."),"error")
        else:
            user=User(name=name,email=email); user.set_password(password); db.session.add(user); db.session.commit(); login_user(user); return redirect(url_for("dashboard.index"))
    return render_template("register.html")
@auth_bp.post("/logout")
def logout(): logout_user(); return redirect(url_for("auth.login"))
