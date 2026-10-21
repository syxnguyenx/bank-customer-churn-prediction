"""Shared churn inference. Importing this module loads no model or Streamlit UI."""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
THRESHOLD = 0.28
INPUT_COLUMNS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
NUMERIC_LIMITS = {
    "CreditScore": (300, 850),
    "Age": (18, 100),
    "Tenure": (0, 10),
    "Balance": (0, None),
    "NumOfProducts": (1, 4),
    "HasCrCard": (0, 1),
    "IsActiveMember": (0, 1),
    "EstimatedSalary": (0, None),
}
EXPECTED_FEATURE_COLUMNS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
    "IsActiveMember", "EstimatedSalary", "Geography_Germany", "Geography_Spain",
    "Gender_Male",
]
BUNDLE_FORMAT = "bank-churn-inference"
BUNDLE_VERSION = 2


def preprocessing_schema():
    """Describe the existing transform; this does not fit any preprocessing."""
    return {
        "version": 1,
        "input_columns": list(INPUT_COLUMNS),
        "numeric_limits": dict(NUMERIC_LIMITS),
        "integer_columns": ["Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember"],
        "categorical_values": {
            "Geography": ["France", "Germany", "Spain"],
            "Gender": ["Female", "Male"],
        },
        "one_hot_columns": {
            "Geography_Germany": ("Geography", "Germany"),
            "Geography_Spain": ("Geography", "Spain"),
            "Gender_Male": ("Gender", "Male"),
        },
        "reference_categories": {"Geography": "France", "Gender": "Female"},
        "missing_values": "reject",
        "non_finite_numbers": "reject",
        "extra_input_columns": "ignore during encoding",
        "scaler": None,
        "training_preprocessing_verified": False,
        "feature_order": list(EXPECTED_FEATURE_COLUMNS),
        "risk_levels": {"low_below": 0.30, "medium_below": 0.60},
    }


def load_artifacts(artifact_dir=APP_DIR):
    """Load the original trusted artifacts, without fitting or saving anything."""
    import joblib

    artifact_dir = Path(artifact_dir)
    model = joblib.load(artifact_dir / "best_xgb.joblib")
    feature_columns = joblib.load(artifact_dir / "feature_columns.joblib")
    return _validate_artifacts(model, feature_columns)


def _validate_artifacts(model, feature_columns):
    """Apply the same artifact checks to file-backed and embedded models."""
    if not isinstance(feature_columns, (list, tuple)):
        raise TypeError("feature_columns.joblib phải chứa danh sách tên cột.")
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Danh sách feature có tên cột trùng lặp.")
    model_feature_count = getattr(model, "n_features_in_", len(feature_columns))
    if model_feature_count != len(feature_columns):
        raise ValueError(
            f"Model cần {model_feature_count} feature nhưng artifact có "
            f"{len(feature_columns)} cột."
        )
    if not hasattr(model, "predict_proba") or len(getattr(model, "classes_", [])) != 2:
        raise TypeError("Model phải hỗ trợ predict_proba cho phân loại nhị phân.")
    if list(feature_columns) != EXPECTED_FEATURE_COLUMNS:
        raise ValueError("Feature artifact không khớp 11 feature và thứ tự hiện tại.")
    return model, list(feature_columns)


def encode_customers(data, feature_columns):
    """Validate and encode raw rows using the original app rules."""
    import numpy as np
    import pandas as pd

    missing = [column for column in INPUT_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("Thiếu cột bắt buộc: " + ", ".join(missing))

    customers = data[INPUT_COLUMNS].copy()
    errors = []
    for column in INPUT_COLUMNS:
        if customers[column].isna().any():
            errors.append(f"{column}: có giá trị trống.")

    for column, (minimum, maximum) in NUMERIC_LIMITS.items():
        values = pd.to_numeric(customers[column], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values)
        if minimum is not None:
            invalid |= values < minimum
        if maximum is not None:
            invalid |= values > maximum
        if column in ("Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember"):
            invalid |= values % 1 != 0
        if invalid.any():
            errors.append(
                f"{column}: cần là số hữu hạn trong khoảng "
                f"{minimum} đến {maximum if maximum is not None else 'không giới hạn'}"
                + (" và là số nguyên." if column in ("Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember") else ".")
            )
        customers[column] = values

    invalid_geography = ~customers["Geography"].isin(["France", "Germany", "Spain"])
    invalid_gender = ~customers["Gender"].isin(["Female", "Male"])
    if invalid_geography.any():
        errors.append("Geography chỉ nhận France, Germany hoặc Spain.")
    if invalid_gender.any():
        errors.append("Gender chỉ nhận Female hoặc Male.")
    if errors:
        raise ValueError("\n".join(errors))

    encoded = pd.DataFrame(index=customers.index)
    for column in NUMERIC_LIMITS:
        encoded[column] = customers[column]
    encoded["Geography_Germany"] = (customers["Geography"] == "Germany").astype(int)
    encoded["Geography_Spain"] = (customers["Geography"] == "Spain").astype(int)
    encoded["Gender_Male"] = (customers["Gender"] == "Male").astype(int)

    unsupported = [column for column in feature_columns if column not in encoded.columns]
    extra = [column for column in encoded.columns if column not in feature_columns]
    if unsupported or extra:
        details = []
        if unsupported:
            details.append("Chưa ánh xạ được: " + ", ".join(unsupported))
        if extra:
            details.append("Feature không có trong artifact: " + ", ".join(extra))
        raise ValueError("; ".join(details))
    return encoded.loc[:, feature_columns]


def predict_customers(features, model):
    """Return class-1 probabilities for an already encoded DataFrame."""
    import numpy as np

    probabilities = np.asarray(model.predict_proba(features))
    classes = list(model.classes_)
    positive_positions = [index for index, label in enumerate(classes) if label == 1]
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes) or len(positive_positions) != 1:
        raise ValueError("Cấu trúc xác suất hoặc nhãn lớp của model không hợp lệ.")
    churn_probability = probabilities[:, positive_positions[0]]
    if not np.isfinite(churn_probability).all() or ((churn_probability < 0) | (churn_probability > 1)).any():
        raise ValueError("Model trả về xác suất không hợp lệ.")
    return churn_probability


def risk_level(probability):
    if probability < 0.30:
        return "Thấp"
    if probability < 0.60:
        return "Trung bình"
    return "Cao"


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(bundle_path):
    """Verify a trusted portable/reference bundle before loading its artifacts."""
    from io import BytesIO

    import joblib

    bundle_path = Path(bundle_path).resolve()
    bundle = joblib.load(bundle_path)
    legacy_reference = (
        bundle.get("format") == "bank-churn-inference-reference"
        and bundle.get("format_version") == 1
    )
    if not legacy_reference and (
        bundle.get("format") != BUNDLE_FORMAT or bundle.get("format_version") != BUNDLE_VERSION
    ):
        raise ValueError("Định dạng inference bundle không được hỗ trợ.")
    if bundle.get("threshold") != THRESHOLD or bundle.get("positive_label") != 1:
        raise ValueError("Threshold hoặc nhãn dương trong bundle không khớp.")
    if bundle.get("preprocessing_schema") != preprocessing_schema():
        raise ValueError("Preprocessing schema trong bundle không khớp module hiện tại.")
    if bundle.get("feature_columns") != EXPECTED_FEATURE_COLUMNS:
        raise ValueError("Feature order trong bundle không khớp module hiện tại.")
    if bundle.get("implementation", {}).get("sha256") != file_sha256(__file__):
        raise ValueError("inference.py đã thay đổi; cần tạo bundle mới để khớp implementation.")
    mode = "reference" if legacy_reference else bundle.get("storage_mode")
    if mode not in ("portable", "reference"):
        raise ValueError("Chế độ lưu trữ bundle không được hỗ trợ.")
    sources = {}
    for key, filename in (("model", "best_xgb.joblib"), ("feature_columns", "feature_columns.joblib")):
        entry = bundle["artifacts"][key]
        if not legacy_reference and entry.get("filename") != filename:
            raise ValueError(f"Tên artifact không hợp lệ: {key}")
        if mode == "portable":
            content = entry.get("content")
            if not isinstance(content, bytes):
                raise ValueError(f"Thiếu dữ liệu nhúng cho artifact: {key}")
            digest = sha256(content).hexdigest()
            sources[key] = BytesIO(content)
        else:
            path = Path(entry["path"])
            path_type = "relative" if legacy_reference else entry.get("path_type")
            if path_type == "relative" and not path.is_absolute() and not path.drive:
                path = (bundle_path.parent / path).resolve()
            elif path_type == "absolute" and path.is_absolute():
                path = path.resolve()
            else:
                raise ValueError(f"Loại đường dẫn artifact không hợp lệ: {key}")
            if path.name != filename:
                raise ValueError(f"Tên file artifact không hợp lệ: {key}")
            digest = file_sha256(path)
            sources[key] = path
        if digest != entry["sha256"]:
            raise ValueError(f"Checksum không khớp cho artifact: {key}")
    # Verify both checksums before deserializing either artifact.
    model = joblib.load(sources["model"])
    feature_columns = joblib.load(sources["feature_columns"])
    return _validate_artifacts(model, feature_columns)


@lru_cache(maxsize=8)
def _cached_artifacts(artifact_dir, bundle_path):
    if bundle_path is not None:
        return load_bundle(bundle_path)
    return load_artifacts(artifact_dir)


def predict_customer(customer_data, *, artifact_dir=None, bundle_path=None):
    """Predict one raw customer mapping; return JSON-compatible Python values.

    Models are loaded lazily and cached per path. Restart the service when deploying
    a new release. Supplying both artifact_dir and bundle_path is ambiguous.
    """
    from collections.abc import Mapping

    import pandas as pd

    if not isinstance(customer_data, Mapping):
        raise TypeError("customer_data phải là mapping của một khách hàng.")
    if artifact_dir is not None and bundle_path is not None:
        raise ValueError("Chỉ chọn artifact_dir hoặc bundle_path.")
    artifact_dir = str(Path(artifact_dir or APP_DIR).resolve())
    bundle_path = str(Path(bundle_path).resolve()) if bundle_path is not None else None
    model, feature_columns = _cached_artifacts(artifact_dir, bundle_path)
    features = encode_customers(pd.DataFrame([dict(customer_data)]), feature_columns)
    probability = float(predict_customers(features, model)[0])
    return {
        "churn_probability": probability,
        "predicted_churn": int(probability >= THRESHOLD),
        "risk_level": risk_level(probability),
        "threshold": THRESHOLD,
    }
