"""Backward-compatible development entry point."""

from run import app


if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"])
