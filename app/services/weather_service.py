"""Weather integration boundary.

No fabricated weather is shown when no provider is configured.
"""

from flask import current_app


def get_weather() -> dict:
    if not current_app.config.get("WEATHER_API_KEY"):
        return {"available": False, "message": "Weather integration is not configured."}
    return {"available": False, "message": "Weather provider integration is pending configuration."}
