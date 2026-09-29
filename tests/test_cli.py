"""Tests de la CLI con el simulador sintético inyectado."""

from __future__ import annotations

import json

import pytest

from conftest import SyntheticWorld
from noshow_guard import cli, simulator


@pytest.fixture(autouse=True)
def synthetic_default(world: SyntheticWorld, monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulador sintético sin registro: la CLI nunca toca data/simulations.db en los tests.
    assert world.simulator.registry is None
    monkeypatch.setattr(simulator, "_DEFAULT", world.simulator)


def test_simulate_existing_patient(
    world: SyntheticWorld, capsys: pytest.CaptureFixture[str]
) -> None:
    patient_id = world.clean["patient_id"].iloc[0]
    code = cli.main(
        [
            "simulate",
            "--patient-id",
            patient_id,
            "--date",
            "2016-06-10",
            "--time",
            "09:30",
            "--as-of",
            "2016-06-01",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["status"] == "ok"
    assert out["lead_time_days"] == 9
    assert out["appointment_time"] == "09:30:00"


def test_simulate_new_patient(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        [
            "simulate",
            "--age",
            "30",
            "--gender",
            "M",
            "--neighbourhood",
            "praia",
            "--scholarship",
            "--date",
            "2016-06-10",
            "--time",
            "08:00",
            "--as-of",
            "2016-06-01",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["features"]["scholarship"] == 1
    assert out["features"]["neighbourhood"] == "PRAIA"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["--age", "30", "--gender", "F", "--neighbourhood", "X", "--date", "2016-05-01"],
            "no se agenda en el pasado",
        ),
        (
            ["--age", "130", "--gender", "F", "--neighbourhood", "X", "--date", "2016-06-10"],
            "Solicitud inválida",
        ),
        (["--patient-id", "999999999", "--date", "2016-06-10"], "no existe"),
    ],
)
def test_errors_exit_with_code_2(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["simulate", *argv, "--time", "10:00", "--as-of", "2016-06-01"])
    assert code == 2
    assert message in capsys.readouterr().err
