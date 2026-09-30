"""Tests del monitoreo de drift: PSI, pruebas, veredicto, lotes y reporte."""

from __future__ import annotations

import dataclasses
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import SyntheticWorld
from noshow_guard.config import DriftConfig, Paths
from noshow_guard.drift import (
    build_report,
    chi2_pvalue,
    compare,
    control_batch,
    evaluate_batches,
    induced_drift_batch,
    psi_categorical,
    psi_numeric,
    psi_status,
    registry_features,
    verdict,
    with_probability,
)
from noshow_guard.features import FEATURE_COLUMNS
from noshow_guard.registry import Registry
from noshow_guard.schemas import SimulationRequest

RNG = np.random.default_rng(0)
CFG = DriftConfig()


# --- PSI -----------------------------------------------------------------------


def test_psi_of_distribution_against_itself_is_zero() -> None:
    x = RNG.normal(size=10_000)
    assert psi_numeric(x, x) == pytest.approx(0.0, abs=1e-12)
    cats = pd.Series(RNG.choice(list("abcd"), 10_000))
    assert psi_categorical(cats, cats) == pytest.approx(0.0, abs=1e-12)


def test_psi_of_independent_sample_from_same_distribution_is_small() -> None:
    assert psi_numeric(RNG.normal(size=20_000), RNG.normal(size=5_000)) < 0.02


def test_psi_detects_mean_shift() -> None:
    assert psi_numeric(RNG.normal(size=10_000), RNG.normal(1.0, 1.0, size=10_000)) > 0.25


def test_psi_gives_repeated_value_its_own_bin() -> None:
    # 95 % ceros en la referencia: pasar a 100 % ceros debe notarse.
    reference = np.r_[np.zeros(9_500), RNG.integers(1, 5, 500)]
    assert psi_numeric(reference, np.zeros(5_000)) > 0.1


def test_psi_handles_new_category() -> None:
    reference = pd.Series(["a"] * 900 + ["b"] * 100)
    current = pd.Series(["a"] * 500 + ["c"] * 500)
    assert psi_categorical(reference, current) > 0.25


def test_chi2_with_single_category_is_one() -> None:
    assert chi2_pvalue(pd.Series(["a"] * 10), pd.Series(["a"] * 10)) == 1.0


@pytest.mark.parametrize(
    ("psi", "expected"),
    [(0.0, "estable"), (0.099, "estable"), (0.10, "vigilar"), (0.25, "vigilar"), (0.26, "alerta")],
)
def test_psi_status_boundaries(psi: float, expected: str) -> None:
    assert psi_status(psi, CFG) == expected


# --- Veredicto -----------------------------------------------------------------


def table(**states: str) -> pd.DataFrame:
    return pd.DataFrame({"variable": list(states), "estado": list(states.values())})


def test_verdict_rules() -> None:
    assert verdict(table(age="alerta"), n=50, cfg=CFG) == "muestra_insuficiente"
    assert verdict(table(age="alerta", diabetes="estable"), n=500, cfg=CFG) == "reentrenar"
    assert verdict(table(age="estable", diabetes="alerta"), n=500, cfg=CFG) == "vigilar"
    assert verdict(table(age="vigilar"), n=500, cfg=CFG) == "vigilar"
    assert verdict(table(age="estable", diabetes="estable"), n=500, cfg=CFG) == "estable"


# --- Lotes con el modelo sintético ----------------------------------------------


@pytest.fixture(scope="module")
def batch_results(world: SyntheticWorld) -> dict[str, str]:
    model, train = world.simulator.model, world.splits["train"]
    reference = with_probability(model, train)
    batches = {
        "control": ("", with_probability(model, control_batch(train, 2000))),
        "drift": ("", with_probability(model, induced_drift_batch(train, 2000))),
    }
    return {r.name: r.verdict for r in evaluate_batches(reference, batches, CFG)}


def test_control_batch_is_stable(batch_results: dict[str, str]) -> None:
    assert batch_results["control"] == "estable"


def test_induced_drift_triggers_retraining_signal(batch_results: dict[str, str]) -> None:
    assert batch_results["drift"] == "reentrenar"


def test_induced_drift_alerts_on_the_modified_variables(world: SyntheticWorld) -> None:
    model, train = world.simulator.model, world.splits["train"]
    result = compare(
        with_probability(model, train),
        with_probability(model, induced_drift_batch(train, 2000)),
        CFG,
    ).set_index("variable")
    assert (result.loc[["lead_time_days", "age", "probability"], "estado"] == "alerta").all()
    assert result.loc["is_male", "estado"] == "estable"


# --- Registro y reporte --------------------------------------------------------


def test_registry_features_only_ok_simulations(world: SyntheticWorld, tmp_path: Path) -> None:
    sim = dataclasses.replace(world.simulator, registry=Registry(tmp_path / "sims.db"))
    base = {"new_patient": {"age": 30, "gender": "F", "neighbourhood": "CENTRO"},
            "appointment_time": time(9, 0), "as_of": date(2016, 6, 1)}  # fmt: skip
    for day in (date(2016, 6, 1), date(2016, 6, 10), date(2016, 6, 20)):  # la 1.ª fuera de alcance
        sim.simulate(SimulationRequest.model_validate({**base, "appointment_date": day}))
    assert sim.registry is not None
    features = registry_features(sim.registry.read())
    assert len(features) == 2
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert features["lead_time_days"].tolist() == [9, 19]


def test_build_report_writes_markdown_and_figures(world: SyntheticWorld, tmp_path: Path) -> None:
    paths = Paths(root=tmp_path)
    results = build_report(
        world.simulator.model,
        "sintetico",
        world.splits["train"],
        world.splits["test"],
        pd.DataFrame(),
        DriftConfig(batch_size=1500),
        paths,
    )
    text = paths.drift_report.read_text(encoding="utf-8")
    verdicts = {r.name: r.verdict for r in results}
    assert verdicts["simulaciones registradas"] == "muestra_insuficiente"
    assert verdicts["control sin drift"] == "estable"
    assert verdicts["drift inducido"] == "reentrenar"
    assert "| drift inducido |" in text and "**reentrenar**" in text
    assert (paths.figures / "drift_psi.png").exists()
    assert (paths.figures / "drift_probability.png").exists()
