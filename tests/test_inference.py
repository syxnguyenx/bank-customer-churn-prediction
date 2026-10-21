"""Regression tests against the original app logic; no Streamlit server needed."""

import importlib.util
import json
import ntpath
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import inference
from package_inference import _reference_path, build_bundle


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "legacy_app_core", ROOT / "tests" / "fixtures" / "legacy_app_core.py"
)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

DEFAULT_CUSTOMER = {
    "CreditScore": 650, "Geography": "France", "Gender": "Female", "Age": 40,
    "Tenure": 5, "Balance": 75000.0, "NumOfProducts": 2, "HasCrCard": 1,
    "IsActiveMember": 1, "EstimatedSalary": 100000.0,
}


def row(**changes):
    return pd.DataFrame([{**DEFAULT_CUSTOMER, **changes}])


def valid_rows():
    yield "default", row()
    for geography in ("France", "Germany", "Spain"):
        for gender in ("Female", "Male"):
            yield f"{geography}/{gender}", row(Geography=geography, Gender=gender)
    for column, limits in inference.NUMERIC_LIMITS.items():
        for value in limits:
            if value is not None:
                yield f"{column}={value}", row(**{column: value})


def invalid_rows():
    yield "missing column", row().drop(columns=["Age"])
    for column in inference.INPUT_COLUMNS:
        yield f"missing {column}", row(**{column: None})
    for column, (minimum, maximum) in inference.NUMERIC_LIMITS.items():
        for value in (np.nan, np.inf, -np.inf, "not-a-number"):
            yield f"{column}={value}", row(**{column: value})
        yield f"{column} below minimum", row(**{column: minimum - 1})
        if maximum is not None:
            yield f"{column} above maximum", row(**{column: maximum + 1})
    for column in ("Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember"):
        value = inference.NUMERIC_LIMITS[column][0] + 0.5
        yield f"fractional {column}", row(**{column: value})
    yield "unsupported geography", row(Geography="Italy")
    yield "unsupported gender", row(Gender="Other")


class InferenceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_hashes = {
            name: inference.file_sha256(ROOT / name)
            for name in ("best_xgb.joblib", "feature_columns.joblib")
        }
        legacy.model = joblib.load(ROOT / "best_xgb.joblib")
        legacy.feature_columns = list(joblib.load(ROOT / "feature_columns.joblib"))
        cls.model, cls.columns = inference.load_artifacts(ROOT)

    @classmethod
    def tearDownClass(cls):
        for name, digest in cls.original_hashes.items():
            if inference.file_sha256(ROOT / name) != digest:
                raise AssertionError(f"Original artifact changed during tests: {name}")

    def test_import_has_no_streamlit_or_artifact_loading(self):
        code = (
            "import sys, inference; "
            "assert 'streamlit' not in sys.modules; "
            "assert 'shap' not in sys.modules; "
            "assert 'joblib' not in sys.modules; "
            "assert inference._cached_artifacts.cache_info().currsize == 0"
        )
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)

    def test_default_feature_names_order_and_values(self):
        encoded = inference.encode_customers(row(), self.columns)
        self.assertEqual(encoded.shape, (1, 11))
        self.assertEqual(list(encoded.columns), legacy.feature_columns)
        self.assertEqual(encoded.iloc[0].tolist(), [650, 40, 5, 75000, 2, 1, 1, 100000, 0, 0, 0])

    def test_all_valid_cases_match_original_encoding_and_probabilities(self):
        for name, data in valid_rows():
            with self.subTest(case=name):
                before = data.copy(deep=True)
                old_features = legacy.encode_customers(data)
                new_features = inference.encode_customers(data, self.columns)
                pd.testing.assert_frame_equal(new_features, old_features, check_exact=True)
                pd.testing.assert_frame_equal(data, before, check_exact=True)
                old_probability = legacy.predict_customers(old_features)
                new_probability = inference.predict_customers(new_features, self.model)
                np.testing.assert_array_equal(new_probability, old_probability)
                self.assertEqual(
                    (new_probability >= inference.THRESHOLD).tolist(),
                    (old_probability >= legacy.THRESHOLD).tolist(),
                )

    def test_batch_preserves_index_extra_columns_and_predictions(self):
        data = pd.concat([data for _, data in valid_rows()], ignore_index=True)
        data.index = range(100, 100 + len(data))
        data["CustomerId"] = [f"sample-{i}" for i in range(len(data))]
        old = legacy.encode_customers(data)
        new = inference.encode_customers(data, self.columns)
        pd.testing.assert_frame_equal(new, old, check_exact=True)
        np.testing.assert_array_equal(
            inference.predict_customers(new, self.model), legacy.predict_customers(old)
        )

    def test_invalid_cases_preserve_error_messages(self):
        for name, data in invalid_rows():
            with self.subTest(case=name):
                with self.assertRaises(ValueError) as old_error:
                    legacy.encode_customers(data)
                with self.assertRaises(ValueError) as new_error:
                    inference.encode_customers(data, self.columns)
                self.assertEqual(str(new_error.exception), str(old_error.exception))

    def test_predict_customer_matches_original_and_is_json_serializable(self):
        result = inference.predict_customer(DEFAULT_CUSTOMER, artifact_dir=ROOT)
        old_probability = float(legacy.predict_customers(legacy.encode_customers(row()))[0])
        self.assertEqual(result, {
            "churn_probability": old_probability,
            "predicted_churn": int(old_probability >= legacy.THRESHOLD),
            "risk_level": legacy.risk_level(old_probability),
            "threshold": 0.28,
        })
        json.dumps(result, allow_nan=False)

    def test_classification_threshold_and_original_risk_boundaries(self):
        self.assertEqual(inference.THRESHOLD, legacy.THRESHOLD)
        for probability in (0.0, 0.279999, 0.28, 0.299999, 0.30, 0.599999, 0.60, 1.0):
            with self.subTest(probability=probability):
                self.assertEqual(inference.risk_level(probability), legacy.risk_level(probability))
                class FixedModel:
                    classes_ = np.array([0, 1])

                    def predict_proba(self, features):
                        return np.array([[1 - probability, probability]])

                with patch.object(inference, "_cached_artifacts", return_value=(FixedModel(), self.columns)):
                    result = inference.predict_customer(DEFAULT_CUSTOMER)
                self.assertEqual(result["predicted_churn"], int(probability >= 0.28))

    def test_bundle_round_trip_and_exclusive_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "inference_bundle.joblib"
            build_bundle(ROOT, output)
            bundle = joblib.load(output)
            self.assertEqual(bundle["storage_mode"], "portable")
            for key, filename in (("model", "best_xgb.joblib"), ("feature_columns", "feature_columns.joblib")):
                self.assertEqual(bundle["artifacts"][key]["content"], (ROOT / filename).read_bytes())
                self.assertNotIn("path", bundle["artifacts"][key])
            self.assertEqual(bundle["threshold"], 0.28)
            self.assertEqual(bundle["preprocessing_schema"], inference.preprocessing_schema())
            self.assertEqual(bundle["feature_columns"], self.columns)
            self.assertFalse(bundle["metadata"]["model_reserialized"])
            result = inference.predict_customer(DEFAULT_CUSTOMER, bundle_path=output)
            self.assertEqual(result, inference.predict_customer(DEFAULT_CUSTOMER, artifact_dir=ROOT))
            original_bundle = output.read_bytes()
            with self.assertRaises(FileExistsError):
                build_bundle(ROOT, output)
            self.assertEqual(output.read_bytes(), original_bundle)

    def test_portable_bundle_relocates_without_reading_original_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "staging"
            deployment = Path(directory) / "deployment"
            staging.mkdir()
            deployment.mkdir()
            # Portable mode must never try relpath, even when Windows would reject it.
            with patch("package_inference.os.path.relpath", side_effect=ValueError("different drives")) as relative:
                output = build_bundle(ROOT, staging / "portable.joblib")
                relative.assert_not_called()
            relocated = deployment / "portable.joblib"
            shutil.copyfile(output, relocated)
            expected = inference.predict_customer(DEFAULT_CUSTOMER, artifact_dir=ROOT)
            original_load = joblib.load
            artifact_paths = {(ROOT / name).resolve() for name in ("best_xgb.joblib", "feature_columns.joblib")}
            embedded_loads = []

            def load_without_original_files(source, *args, **kwargs):
                if isinstance(source, (str, Path)):
                    if Path(source).resolve() in artifact_paths:
                        raise AssertionError("Portable loading tried to read an original artifact")
                else:
                    embedded_loads.append(source)
                return original_load(source, *args, **kwargs)

            with patch("joblib.load", side_effect=load_without_original_files):
                actual = inference.predict_customer(DEFAULT_CUSTOMER, bundle_path=relocated)
            self.assertEqual(actual, expected)
            self.assertEqual(len(embedded_loads), 2)

    def test_windows_different_drive_reference_path(self):
        # Use Windows path semantics on every OS; no physical extra drive is needed.
        artifact = PureWindowsPath("X:/trained/best_xgb.joblib")
        destination = PureWindowsPath("Y:/release")
        windows_relpath = ntpath.relpath
        with self.assertRaises(ValueError):
            windows_relpath(str(artifact), str(destination))
        with patch("package_inference.os.path.relpath", side_effect=windows_relpath):
            reference = _reference_path(artifact, destination)
        self.assertEqual(reference["path_type"], "absolute")
        self.assertEqual(PureWindowsPath(reference["path"]), artifact)

    def test_reference_bundle_cross_drive_fallback_loads_and_checks_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "external.joblib"
            # Exercise the real fallback while referencing the real model files.
            with patch("package_inference.os.path.relpath", side_effect=ValueError("different drives")):
                build_bundle(ROOT, output, mode="reference")
            bundle = joblib.load(output)
            self.assertEqual(bundle["storage_mode"], "reference")
            for entry in bundle["artifacts"].values():
                self.assertEqual(entry["path_type"], "absolute")
                self.assertTrue(Path(entry["path"]).is_absolute())
            model, columns = inference.load_bundle(output)
            encoded = inference.encode_customers(row(), columns)
            np.testing.assert_array_equal(
                inference.predict_customers(encoded, model),
                legacy.predict_customers(legacy.encode_customers(row())),
            )
            bundle["artifacts"]["model"]["sha256"] = "0" * 64
            changed = Path(directory) / "bad-checksum.joblib"
            joblib.dump(bundle, changed)
            with self.assertRaisesRegex(ValueError, "Checksum"):
                inference.load_bundle(changed)

    def test_reference_bundle_same_drive_keeps_relative_paths(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = build_bundle(ROOT, Path(directory) / "relative.joblib", mode="reference")
            bundle = joblib.load(output)
            for entry in bundle["artifacts"].values():
                self.assertEqual(entry["path_type"], "relative")
                self.assertFalse(Path(entry["path"]).is_absolute())
            model, columns = inference.load_bundle(output)
            np.testing.assert_array_equal(
                inference.predict_customers(inference.encode_customers(row(), columns), model),
                legacy.predict_customers(legacy.encode_customers(row())),
            )

    def test_bundle_rejects_changed_schema_threshold_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            output = build_bundle(ROOT, Path(directory) / "bundle.joblib")
            original = joblib.load(output)
            for changed_field in ("threshold", "schema", "feature_order", "checksum", "content", "implementation"):
                with self.subTest(field=changed_field):
                    import copy
                    altered = copy.deepcopy(original)
                    if changed_field == "threshold":
                        altered["threshold"] = 0.50
                    elif changed_field == "schema":
                        altered["preprocessing_schema"]["scaler"] = "unexpected"
                    elif changed_field == "checksum":
                        altered["artifacts"]["model"]["sha256"] = "0" * 64
                    elif changed_field == "feature_order":
                        altered["feature_columns"] = list(reversed(altered["feature_columns"]))
                    elif changed_field == "content":
                        altered["artifacts"]["model"]["content"] += b"corruption"
                    else:
                        altered["implementation"]["sha256"] = "0" * 64
                    bad_path = Path(directory) / f"bad-{changed_field}.joblib"
                    joblib.dump(altered, bad_path)
                    with self.assertRaises(ValueError):
                        inference.load_bundle(bad_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
