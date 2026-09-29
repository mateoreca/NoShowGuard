"""Tests de métricas, política por costo y segmentos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from noshow_guard.config import CostConfig
from noshow_guard.evaluation import (
    Thresholds,
    action_costs,
    analytic_thresholds,
    assign_actions,
    binary_metrics,
    optimize_thresholds,
    risk_cutoffs,
    risk_level,
    score_metrics,
    segment_table,
)

COSTS = CostConfig()


def test_analytic_thresholds_default_costs() -> None:
    th = analytic_thresholds(COSTS)
    assert th.standard == pytest.approx(1 / 3)  # 1 / (20 * 0.15)
    assert th.reinforced == pytest.approx(2 / 3)  # (3 - 1) / (20 * (0.30 - 0.15))


def test_unprofitable_reminders_give_threshold_one() -> None:
    # Hueco = 10, recordatorio = 1, efecto 10 %: el ahorro esperado (10 * p * 0.1 = p) nunca
    # supera el costo (1), así que nunca conviene recordar.
    costs = CostConfig(no_show_cost=10, standard_effect=0.10, reinforced_effect=0.20)
    th = analytic_thresholds(costs)
    assert th.standard == 1.0
    assert th.reinforced == 1.0


@pytest.mark.parametrize(
    ("p", "expected"),
    [(0.0, 0), (0.3299, 0), (0.33, 1), (0.6699, 1), (0.67, 2), (1.0, 2)],
)
def test_assign_actions_boundaries(p: float, expected: int) -> None:
    assert assign_actions(np.array([p]), Thresholds(0.33, 0.67))[0] == expected


def test_action_costs_toy_example() -> None:
    actions = np.array([0, 0, 1, 2])
    y = np.array([0, 1, 1, 1])
    # Sin acción y asiste: 0. Sin acción y falta: 20.
    # Estándar y falta: 1 + 20 * 0.85. Reforzado y falta: 3 + 20 * 0.7.
    np.testing.assert_allclose(action_costs(actions, y, COSTS), [0, 20, 18, 17])


def test_optimized_thresholds_match_analytic_when_calibrated() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = rng.binomial(1, p)  # probabilidades perfectamente calibradas
    th, table = optimize_thresholds(p, y, COSTS)
    ref = analytic_thresholds(COSTS)
    assert th.standard == pytest.approx(ref.standard, abs=0.03)
    assert th.reinforced == pytest.approx(ref.reinforced, abs=0.03)
    assert (table["reinforced"] >= table["standard"]).all()


def test_risk_cutoffs_are_quantiles_of_reference() -> None:
    cutoffs = risk_cutoffs(np.arange(101) / 100)
    assert cutoffs == {"medio": pytest.approx(0.5), "alto": pytest.approx(0.9)}


@pytest.mark.parametrize(
    ("p", "expected"), [(0.1, "bajo"), (0.28, "medio"), (0.3, "medio"), (0.37, "alto")]
)
def test_risk_level_boundaries(p: float, expected: str) -> None:
    assert risk_level(p, {"medio": 0.28, "alto": 0.37}) == expected


def test_binary_metrics() -> None:
    m = binary_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert (m["precision"], m["recall"], m["fpr"], m["flagged_rate"]) == (0.5, 0.5, 0.5, 0.5)


def test_score_metrics_skip_probability_metrics_for_raw_scores() -> None:
    m = score_metrics(np.array([0, 1, 0, 1]), np.array([1.0, 5.0, 2.0, 9.0]))
    assert m["roc_auc"] == 1.0
    assert "brier" not in m


def test_segment_table_one_row_per_group() -> None:
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.8, 0.2, 0.6, 0.4, 0.5])
    table = segment_table(y, p, pd.Series(["a", "a", "a", "b", "b", "b"]), threshold=0.5)
    assert table["group"].tolist() == ["a", "b"]
    assert table["n"].tolist() == [3, 3]
    assert table.loc[0, "recall"] == 1.0
