"""Tests de acciones, plantillas, MockSender, flujo completo, what-if e impacto."""

from __future__ import annotations

import dataclasses
import io
import json
import socket
from datetime import date, time
from pathlib import Path

import numpy as np
import pytest

from conftest import SyntheticWorld
from noshow_guard import cli, simulator
from noshow_guard.actions import (
    ALLOWED_PLACEHOLDERS,
    Message,
    MockSender,
    decide_action,
    placeholders,
    render_message,
    spanish_date,
    templates,
    thresholds_from_metadata,
)
from noshow_guard.config import CostConfig, ImpactConfig, ImpactScenario
from noshow_guard.evaluation import ACTIONS, Thresholds, assign_actions
from noshow_guard.impact import evaluate_policy, impact_table, render_markdown
from noshow_guard.registry import Registry
from noshow_guard.schemas import SimulationRequest
from noshow_guard.simulator import Simulator

TH = Thresholds(standard=0.36, reinforced=0.67)
NEW = {"age": 30, "gender": "F", "neighbourhood": "CENTRO"}


def req(**overrides: object) -> SimulationRequest:
    payload: dict[str, object] = {
        "new_patient": NEW,
        "appointment_date": date(2016, 6, 14),
        "appointment_time": time(8, 30),
        "as_of": date(2016, 6, 1),
    }
    payload.update(overrides)
    return SimulationRequest.model_validate(payload)


# --- Decisión ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.0, "sin_accion"),
        (0.3599, "sin_accion"),
        (0.36, "recordatorio_estandar"),
        (0.6699, "recordatorio_estandar"),
        (0.67, "recordatorio_reforzado_con_confirmacion"),
        (1.0, "recordatorio_reforzado_con_confirmacion"),
    ],
)
def test_decide_action_boundaries(p: float, expected: str) -> None:
    assert decide_action(p, TH) == expected


def test_decide_action_is_pure_and_matches_batch_rule() -> None:
    grid = np.linspace(0, 1, 201)
    batch = [ACTIONS[i] for i in assign_actions(grid, TH)]
    assert [decide_action(float(p), TH) for p in grid] == batch
    assert [decide_action(0.5, TH) for _ in range(3)] == ["recordatorio_estandar"] * 3


def test_thresholds_come_from_metadata() -> None:
    meta = {"thresholds": {"standard": 0.2, "reinforced": 0.5}}
    assert thresholds_from_metadata(meta) == Thresholds(0.2, 0.5)


# --- Plantillas ----------------------------------------------------------------


def test_templates_cover_actions_and_only_use_date_and_time() -> None:
    assert set(templates()) == set(ACTIONS[1:])
    for text in templates().values():
        assert placeholders(text) <= ALLOWED_PLACEHOLDERS


def test_render_message() -> None:
    assert render_message("sin_accion", date(2016, 6, 14), time(8, 30)) is None
    text = render_message("recordatorio_estandar", date(2016, 6, 14), time(8, 30))
    assert text is not None and "martes 14 de junio de 2016" in text and "08:30" in text
    reinforced = render_message(
        "recordatorio_reforzado_con_confirmacion", date(2016, 6, 14), time(8, 30)
    )
    assert reinforced is not None and "SÍ" in reinforced


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2016, 6, 14), "martes 14 de junio de 2016"),
        (date(2026, 1, 3), "sábado 3 de enero de 2026"),
    ],
)
def test_spanish_date(day: date, expected: str) -> None:
    assert spanish_date(day) == expected


# --- MockSender ------------------------------------------------------------------


def test_mock_sender_records_and_never_uses_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("MockSender intentó usar la red")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    stream = io.StringIO()
    sender = MockSender(stream=stream)
    message = Message("recordatorio_estandar", "hola", date(2016, 6, 14), time(8, 30))
    sender.send(message)
    assert sender.outbox == [message]
    assert "NO enviado" in stream.getvalue()


# --- Flujo completo: paciente -> simulación -> acción -> mensaje -----------------------


def with_thresholds(world: SyntheticWorld, tmp_path: Path, standard: float) -> Simulator:
    metadata = {
        **world.simulator.metadata,
        "thresholds": {"standard": standard, "reinforced": 0.99},
    }
    return dataclasses.replace(
        world.simulator,
        metadata=metadata,
        registry=Registry(tmp_path / "sims.db"),
        sender=MockSender(),
    )


def test_full_flow_with_reminder(world: SyntheticWorld, tmp_path: Path) -> None:
    sim = with_thresholds(world, tmp_path, standard=0.0)  # toda cita recibe recordatorio
    result = sim.simulate(req())
    assert result.action == "recordatorio_estandar"
    assert result.message_preview == render_message(
        "recordatorio_estandar", date(2016, 6, 14), time(8, 30)
    )
    assert isinstance(sim.sender, MockSender)
    assert [m.text for m in sim.sender.outbox] == [result.message_preview]
    assert sim.registry is not None
    assert sim.registry.read()["action"].tolist() == ["recordatorio_estandar"]


def test_full_flow_without_action_sends_nothing(world: SyntheticWorld, tmp_path: Path) -> None:
    sim = with_thresholds(world, tmp_path, standard=1.0)
    result = sim.simulate(req())
    assert (result.action, result.message_preview) == ("sin_accion", None)
    assert isinstance(sim.sender, MockSender) and sim.sender.outbox == []
    assert sim.registry is not None
    assert sim.registry.read()["action"].tolist() == ["sin_accion"]


def test_out_of_scope_has_no_action(world: SyntheticWorld, tmp_path: Path) -> None:
    sim = with_thresholds(world, tmp_path, standard=0.0)
    result = sim.simulate(req(appointment_date=date(2016, 6, 1)))
    assert result.action is None and result.message_preview is None
    assert isinstance(sim.sender, MockSender) and sim.sender.outbox == []


# --- What-if ---------------------------------------------------------------------


def test_what_if_one_row_per_lead_time(world: SyntheticWorld, tmp_path: Path) -> None:
    sim = with_thresholds(world, tmp_path, standard=0.0)
    result = sim.what_if(req(), [14, 1, 7, 3])
    assert [r.lead_time_days for r in result.rows] == [1, 3, 7, 14]
    assert [r.as_of for r in result.rows] == [
        date(2016, 6, 13), date(2016, 6, 11), date(2016, 6, 7), date(2016, 5, 31)
    ]  # fmt: skip
    assert all(r.status == "ok" and r.probability_no_show is not None for r in result.rows)
    assert any("NO un efecto causal" in n for n in result.notes)
    # Análisis puro: no registra ni envía.
    assert sim.registry is not None and sim.registry.read().empty
    assert isinstance(sim.sender, MockSender) and sim.sender.outbox == []


def test_what_if_edge_cases(world: SyntheticWorld) -> None:
    rows = world.simulator.what_if(req(), [0, 1]).rows
    assert rows[0].status == "fuera_de_alcance" and rows[0].probability_no_show is None
    with pytest.raises(ValueError, match=">= 0"):
        world.simulator.what_if(req(), [-1])
    old = world.simulator.what_if(
        req(new_patient=None, patient_id=str(world.clean["patient_id"].iloc[0]),
            appointment_date=date(2016, 6, 10)),
        [400],
    ).rows  # fmt: skip
    assert old[0].status == "error" and "no tiene registros" in (old[0].detail or "")


def test_cli_what_if_with_plot(
    world: SyntheticWorld, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    monkeypatch.setattr(simulator, "_DEFAULT", world.simulator)
    plot = tmp_path / "what_if.png"
    code = cli.main(
        ["what-if", "--age", "30", "--gender", "F", "--neighbourhood", "CENTRO",
         "--date", "2016-06-14", "--time", "08:30", "--lead-times", "1", "3", "7", "14",
         "--plot", str(plot)]
    )  # fmt: skip
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert [r["lead_time_days"] for r in out["rows"]] == [1, 3, 7, 14]
    assert plot.exists()


# --- Impacto ---------------------------------------------------------------------

COSTS = CostConfig()
Y = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 0])
P = np.array([0.9, 0.1, 0.5, 0.4, 0.2, 0.1, 0.3, 0.8, 0.2, 0.1])


def test_evaluate_policy_counts() -> None:
    nothing = evaluate_policy(np.zeros(10, dtype=int), Y, COSTS)
    assert (nothing["no_shows_evitados"], nothing["costo_total"]) == (0.0, 4 * 20.0)
    everyone = evaluate_policy(np.ones(10, dtype=int), Y, COSTS)
    assert everyone["recordatorios_estandar"] == 10
    assert everyone["no_shows_evitados"] == pytest.approx(4 * 0.15)
    assert everyone["costo_total"] == pytest.approx(10 * 1 + 20 * 4 * 0.85)


def test_impact_table_policies_and_scenarios() -> None:
    impact = ImpactConfig(
        scenarios=(ImpactScenario("bajo", 0.05, 0.1), ImpactScenario("alto", 0.5, 0.6))
    )
    table = impact_table(P, Y, TH, COSTS, impact)
    assert len(table) == 6
    nothing = table[table["politica"] == "no_hacer_nada"]
    assert (nothing["ahorro_vs_nada"] == 0).all() and (nothing["recordatorios"] == 0).all()
    everyone = table[table["politica"] == "recordar_a_todos"].set_index("escenario")
    assert everyone.loc["alto", "ahorro_vs_nada"] > everyone.loc["bajo", "ahorro_vs_nada"]
    model = table[table["politica"] == "segun_modelo"].iloc[0]
    assert model["recordatorios"] == int((TH.standard <= P).sum())


def test_model_policy_with_unreachable_thresholds_equals_nothing() -> None:
    table = impact_table(P, Y, Thresholds(2.0, 2.0), COSTS)
    model = table[table["politica"] == "segun_modelo"]
    assert (model["recordatorios"] == 0).all() and (model["ahorro_vs_nada"] == 0).all()


def test_adjusted_thresholds_are_chosen_on_validation_only() -> None:
    rng = np.random.default_rng(1)
    p_val = rng.uniform(0, 1, 20_000)
    y_val = rng.binomial(1, p_val)  # calibradas: el óptimo en val es el analítico
    impact = ImpactConfig(scenarios=(ImpactScenario("alto", 0.25, 0.45),))
    table = impact_table(P, Y, TH, COSTS, impact, validation=(p_val, y_val))
    adjusted = table[table["politica"] == "segun_modelo_ajustado"].iloc[0]
    # Con efecto 25 %: estándar conviene si p > 1 / (20 * 0,25) = 0,20.
    assert adjusted["umbral_estandar"] == pytest.approx(0.20, abs=0.03)
    fixed = table[table["politica"] == "segun_modelo"].iloc[0]
    assert fixed["umbral_estandar"] == TH.standard
    # Sin datos de validación, la política ajustada no aparece.
    assert "segun_modelo_ajustado" not in set(impact_table(P, Y, TH, COSTS, impact)["politica"])


def test_impact_report_is_labeled_as_simulation() -> None:
    text = render_markdown(impact_table(P, Y, TH, COSTS), TH, COSTS, len(Y), "v")
    assert "SIMULACIÓN, NO RESULTADO REAL" in text
    assert "Supuestos de efecto" in text and "pesimista" in text and "optimista" in text
