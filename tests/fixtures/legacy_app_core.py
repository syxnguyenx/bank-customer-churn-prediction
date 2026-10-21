"""Frozen pre-refactor logic, used only as the regression test oracle."""

import numpy as np
import pandas as pd


THRESHOLD = 0.28


INPUT_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
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


def encode_customers(data):
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


def predict_customers(features):
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
