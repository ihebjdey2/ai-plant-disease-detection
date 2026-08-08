"""Convert the reviewed companion CSV into repository-owned JSON metadata."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main(source: Path, destination: Path) -> None:
    with source.open(encoding="cp1252", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 39:
        raise ValueError(f"Expected 39 disease records, found {len(rows)}")
    metadata = []
    for row in rows:
        metadata.append({
            "class_index": int(row["index"]),
            "disease_name": row["disease_name"].strip(),
            "description": row["description"].strip(),
            "recommended_steps": row["Possible Steps"].strip(),
            "reference_image_url": row["image_url"].strip(),
            "source": "Companion Plant-Disease-Detection disease_info.csv",
        })
    destination.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
