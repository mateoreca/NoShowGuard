"""Entrenamiento de humo con datos sintéticos, calibración y SHAP."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from noshow_guard.config import SearchConfig
from noshow_guard.explain import top_factors
from noshow_guard.features import FEATURE_COLUMNS
from noshow_guard.model import fit_calibrator, load_model
from noshow_guard.train import TrainResult, _json_safe, fit_all, rule_lead_threshold

SMOKE = SearchConfig(n_iter=2, max_estimators=60, early_stopping_rounds=10, calibration_folds=3)


@pytest.fixture(scope="module")
def result(synthetic_splits: dict[str, pd.DataFrame]) -> TrainResult:
    return fit_all(synthetic_splits, search=SMOKE)


def test_smoke_training_produces_valid_probabilities(
    result: TrainResult, synthetic_splits: dict[str, pd.DataFrame]
) -> None:
    p = result.model.predict_proba(synthetic_splits["test"])
    assert p.shape == (len(synthetic_splits["test"]),)
    assert np.all((p >= 0) & (p <= 1))
    # Con señal sintética clara, el modelo debe superar a la clase mayoritaria.
    test = result.metrics["test"]
    assert test["regresion_logistica_calibrada"]["pr_auc"] > test["clase_mayoritaria"]["pr_auc"]


def test_metrics_contain_required_sections(result: TrainResult) -> None:
    for key in ("thresholds", "calibration", "segments_test", "shap", "train_ranges"):
        assert key in result.metrics
    assert result.thresholds.standard <= result.thresholds.reinforced
    assert set(result.metrics["segments_test"]["historial"][0]) >= {"group", "recall", "fpr"}


def test_principal_model_is_calibrated_logistic_regression(result: TrainResult) -> None:
    assert isinstance(result.model.pipeline.named_steps["clf"], LogisticRegression)
    assert result.metrics["principal_model"] == "regresion_logistica_calibrada"
    assert result.metrics["model_decision"]["decidido_despues_de_ver_test"] is True
    # LightGBM sigue en la tabla como comparación.
    assert "lightgbm_calibrado" in result.metrics["test"]


def test_rolling_origin_comparison_does_not_use_test(result: TrainResult) -> None:
    rows = result.metrics["rolling_origin_pr_auc"]
    assert len(rows) == 3
    assert all(row["evalua_hasta"] <= "2016-05-31" for row in rows)
    assert all({"regresion_logistica", "lightgbm"} <= set(row) for row in rows)


def test_load_model_uses_metadata_file(
    result: TrainResult, synthetic_splits: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    joblib.dump(result.model, tmp_path / "v1.joblib")
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps({"model_file": "v1.joblib", "thresholds": {}}), "utf-8")
    loaded, meta = load_model(meta_path)
    test = synthetic_splits["test"]
    np.testing.assert_array_equal(loaded.predict_proba(test), result.model.predict_proba(test))
    assert meta["model_file"] == "v1.joblib"


def test_load_model_rejects_wrong_artifact(tmp_path: Path) -> None:
    joblib.dump({"not": "a model"}, tmp_path / "bad.joblib")
    (tmp_path / "metadata.json").write_text(json.dumps({"model_file": "bad.joblib"}), "utf-8")
    with pytest.raises(TypeError, match="NoShowModel"):
        load_model(tmp_path / "metadata.json")


def test_shap_contributions_are_additive(
    result: TrainResult, synthetic_splits: dict[str, pd.DataFrame]
) -> None:
    rows = synthetic_splits["test"].head(50)
    contrib = result.model.contributions(rows)
    assert list(contrib.columns) == [*FEATURE_COLUMNS, "base_value"]
    raw = result.model.raw_proba(rows)
    np.testing.assert_allclose(contrib.sum(axis=1), np.log(raw / (1 - raw)), atol=1e-6)


def test_changing_one_feature_changes_only_its_contribution(
    result: TrainResult, synthetic_splits: dict[str, pd.DataFrame]
) -> None:
    row = synthetic_splits["test"].iloc[[0]].copy()
    other = row.assign(age=95 if row["age"].iloc[0] < 50 else 5)
    diff = result.model.contributions(other) - result.model.contributions(row)
    changed = diff.columns[diff.iloc[0].abs() > 1e-12].tolist()
    assert changed == ["age"]


def test_top_factors_sorted_by_magnitude(
    result: TrainResult, synthetic_splits: dict[str, pd.DataFrame]
) -> None:
    row = synthetic_splits["test"].iloc[[0]]
    contrib = result.model.contributions(row).iloc[0]
    factors = top_factors(contrib, row.iloc[0])
    assert len(factors) == 3
    magnitudes = [abs(f["shap"]) for f in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(f["effect"] in {"+", "-"} for f in factors)


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibrators_are_monotonic(method: str) -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.6, 2000)
    y = rng.binomial(1, np.clip(p * 0.8, 0, 1))
    cal = fit_calibrator(method, p, y)  # type: ignore[arg-type]
    grid = np.linspace(0.01, 0.99, 50)
    assert np.all(np.diff(cal.predict(grid)) >= -1e-12)


def test_rule_threshold_picks_value_in_range() -> None:
    lead = np.array([1, 2, 10, 20, 30, 40])
    y = np.array([0, 0, 1, 1, 1, 0])
    assert 1 <= rule_lead_threshold(lead, y) <= 60


def test_json_safe_converts_nan_and_numpy() -> None:
    assert _json_safe({"a": float("nan"), "b": np.int64(3), "c": [np.float64(0.5)]}) == {
        "a": None,
        "b": 3,
        "c": [0.5],
    }
