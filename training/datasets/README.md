# Dataset V2 acquisition metadata

This directory stores versioned mappings, manifests, and audit reports. Dataset images, downloads, and local audit logs are intentionally excluded from Git.

## Local layout

```text
training/datasets/
├── downloads/       # ignored source archives
├── raw/             # ignored read-only materializations
├── local-audits/    # ignored execution logs
├── mappings/        # reviewed source-to-deployed taxonomy mappings
├── manifests/       # versioned per-image metadata and hashes
└── reports/         # versioned duplicate and integrity reports
```

PlantDoc TEST belongs under `evaluation/datasets/` and is a locked benchmark. It must never be copied into `training/datasets/raw/` or used for model selection.

## Audited inputs

1. Clone the official PlantDoc repository without checking out Windows-invalid filenames:

   ```powershell
   git clone --no-checkout https://github.com/pratikkayal/PlantDoc-Dataset evaluation/datasets/plantdoc
   ```

   The audit reads the immutable Git tree at `5467f6012d78d1c446145d5f582da6096f852ae8`, even if the clone's current `HEAD` later changes.

2. Download version 1 of the official Potato archive from <https://doi.org/10.17632/d5b3fzpw3g.1> as:

   ```text
   training/datasets/downloads/potato-leaf-disease-d5b3fzpw3g-v1.zip
   ```

   Expected SHA-256:

   ```text
   549c7f3343422fa2b77b6fb2c5009a52215aa00626b2646435ba19f4826f8192
   ```

## Run the audit

```powershell
python scripts/audit_dataset_v2_sources.py
```

The command validates all images, materializes PlantDoc TRAIN and only Potato `orig_*` entries, verifies mappings, and regenerates manifests and duplicate reports. It never removes duplicates or creates train/validation/test splits.

See `docs/dataset-v2-acquisition-audit.md` for the reviewed Step 5B findings.
