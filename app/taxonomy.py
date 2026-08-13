"""Compatibility re-export for Flask application imports."""

from training.taxonomy import CLASS_NAMES


_CROP_ICONS = {
    "Apple": "🍎",
    "Blueberry": "🫐",
    "Cherry": "🍒",
    "Corn": "🌽",
    "Grape": "🍇",
    "Orange": "🍊",
    "Peach": "🍑",
    "Bell pepper": "🫑",
    "Potato": "🥔",
    "Raspberry": "🍓",
    "Soybean": "🌱",
    "Squash": "🎃",
    "Strawberry": "🍓",
    "Tomato": "🍅",
}


def supported_crops() -> list[dict[str, str | int]]:
    """Return the real crop coverage derived from the frozen model taxonomy."""
    return [
        {
            "name": crop,
            "icon": icon,
            "class_count": sum(label.startswith(f"{crop} ") for label in CLASS_NAMES),
        }
        for crop, icon in _CROP_ICONS.items()
    ]
