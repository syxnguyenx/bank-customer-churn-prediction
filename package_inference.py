"""Package original artifact bytes, or explicit external references, with joblib."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import os
from pathlib import Path
import platform

import joblib

import inference


def _reference_path(artifact_path, bundle_dir):
    """Use a relative path when possible; label cross-drive references explicitly."""
    try:
        relative = os.path.relpath(artifact_path, bundle_dir)
    except ValueError:
        # Windows cannot compute relpath across drive letters or UNC shares.
        # The caller supplies a resolved absolute artifact path.
        return {"path": str(artifact_path), "path_type": "absolute"}
    return {"path": Path(relative).as_posix(), "path_type": "relative"}


def build_bundle(artifact_dir=inference.APP_DIR, output=None, *, mode="portable"):
    if mode not in ("portable", "reference"):
        raise ValueError("mode must be 'portable' or 'reference'")
    artifact_dir = Path(artifact_dir).resolve()
    output = Path(output or artifact_dir / "inference_bundle.joblib").resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    feature_columns = joblib.load(artifact_dir / "feature_columns.joblib")
    if not isinstance(feature_columns, (list, tuple)) or list(feature_columns) != inference.EXPECTED_FEATURE_COLUMNS:
        raise ValueError("Feature artifact does not match the current 11-feature schema.")
    artifacts = {}
    for key, filename in (("model", "best_xgb.joblib"), ("feature_columns", "feature_columns.joblib")):
        artifact_path = artifact_dir / filename
        content = artifact_path.read_bytes()
        entry = {"filename": filename, "sha256": sha256(content).hexdigest()}
        if mode == "portable":
            # Preserve the original serialization byte-for-byte. Do not load XGBoost.
            entry["content"] = content
        else:
            entry.update(_reference_path(artifact_path, output.parent))
        artifacts[key] = entry
    dependencies = {}
    for name in ("joblib", "numpy", "pandas", "scikit-learn", "xgboost", "shap", "streamlit", "matplotlib"):
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = None
    bundle = {
        "format": inference.BUNDLE_FORMAT,
        "format_version": inference.BUNDLE_VERSION,
        "storage_mode": mode,
        "threshold": inference.THRESHOLD,
        "positive_label": 1,
        "feature_columns": list(feature_columns),
        "preprocessing_schema": inference.preprocessing_schema(),
        "implementation": {
            "module": "inference",
            "encoder": "inference.encode_customers",
            "entrypoint": "inference.predict_customer",
            "sha256": inference.file_sha256(inference.__file__),
        },
        "artifacts": artifacts,
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "packaging_python": platform.python_version(),
            "packaging_distribution_versions": dependencies,
            "training_versions": None,
            "training_notebook_available": False,
            "scaler_usage_at_training": "unverified; no scaler added",
            "model_reserialized": False,
            "runtime_parity_verified_by_packager": False,
        },
    }
    # Exclusive creation prevents overwriting old artifacts or an existing bundle.
    with output.open("xb") as handle:
        joblib.dump(bundle, handle)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=inference.APP_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("portable", "reference"), default="portable")
    args = parser.parse_args()
    output = build_bundle(args.artifact_dir, args.output, mode=args.mode)
    print(f"Created {args.mode} bundle: {output}")
    if args.mode == "reference":
        print("External references require the referenced files; cross-drive references use absolute paths.")
    print("Original model was not loaded, fitted, or rewritten. Run runtime parity tests separately.")


if __name__ == "__main__":
    main()
