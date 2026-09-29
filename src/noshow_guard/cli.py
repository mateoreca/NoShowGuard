"""Interfaz de línea de comandos del simulador.

Ejemplos:
    python -m noshow_guard.cli simulate --patient-id <PATIENT_ID> --date 2016-06-10 \
        --time 09:30 --as-of 2016-06-01
    python -m noshow_guard.cli simulate --age 30 --gender F --neighbourhood "JARDIM DA PENHA" \
        --date 2026-10-15 --time 08:00 --no-log
    python -m noshow_guard.cli model-info
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from noshow_guard.config import PATHS
from noshow_guard.schemas import SimulationRequest, SimulationResult
from noshow_guard.simulator import SimulationError, simulate

NEW_PATIENT_FLAGS = ("scholarship", "hypertension", "diabetes", "alcoholism")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="noshow_guard.cli", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    sim = commands.add_parser("simulate", help="Simula una cita sin agendarla.")
    sim.add_argument("--date", required=True, help="Fecha de la cita (AAAA-MM-DD).")
    sim.add_argument("--time", required=True, help="Hora de la cita (HH:MM). No es feature.")
    sim.add_argument("--as-of", help="Fecha de agendamiento simulada (por defecto, hoy).")
    sim.add_argument("--patient-id", help="Paciente existente en el dataset.")
    sim.add_argument(
        "--no-log", action="store_true", help="No guardar la simulación en el registro SQLite."
    )

    new = sim.add_argument_group("paciente nuevo (en lugar de --patient-id)")
    new.add_argument("--age", type=int)
    new.add_argument("--gender", choices=["F", "M"])
    new.add_argument("--neighbourhood")
    new.add_argument("--handicap", type=int, default=0)
    for flag in NEW_PATIENT_FLAGS:
        new.add_argument(f"--{flag}", action="store_true")

    commands.add_parser("model-info", help="Versión y metadata del modelo cargado.")
    return parser


def model_info(metadata_path: Path = PATHS.model_metadata) -> dict[str, Any]:
    """Resumen de la metadata del modelo principal (no carga el modelo ni los datos)."""
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "model_version": meta["model_version"],
        "model_file_present": (metadata_path.parent / meta["model_file"]).exists(),
        "created_at": meta["created_at"],
        "principal_model": meta["principal_model"],
        "decided_after_test": meta["model_decision"]["decidido_despues_de_ver_test"],
        "calibration_method": meta["calibration_method"],
        "thresholds": meta["thresholds"],
        "risk_cutoffs": meta["risk_cutoffs"],
        "costs": meta["costs"],
        "costs_are_assumptions": meta["costs_are_assumptions"],
        "test_metrics": meta["test_metrics"],
        "data_hash": meta["data_hash"][:12],
    }


def request_from_args(args: argparse.Namespace) -> SimulationRequest:
    """Construye la solicitud; Pydantic valida fechas, rangos y paciente."""
    payload: dict[str, Any] = {"appointment_date": args.date, "appointment_time": args.time}
    if args.as_of:
        payload["as_of"] = args.as_of
    if args.patient_id:
        payload["patient_id"] = args.patient_id
    if args.age is not None or args.gender or args.neighbourhood:
        payload["new_patient"] = {
            "age": args.age,
            "gender": args.gender,
            "neighbourhood": args.neighbourhood,
            "handicap": args.handicap,
            **{flag: getattr(args, flag) for flag in NEW_PATIENT_FLAGS},
        }
    return SimulationRequest.model_validate(payload)


def render(result: SimulationResult) -> str:
    """JSON legible; la probabilidad se redondea solo para mostrarla."""
    data = result.model_dump(mode="json")
    if data["probability_no_show"] is not None:
        data["probability_no_show"] = round(data["probability_no_show"], 4)
    return json.dumps(data, indent=2, ensure_ascii=False)


def _validation_message(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'solicitud'}: {err['msg']}"
        for err in exc.errors()
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "model-info":
        print(json.dumps(model_info(), indent=2, ensure_ascii=False))
        return 0
    try:
        result = simulate(request_from_args(args), log=not args.no_log)
    except ValidationError as exc:
        print(f"Solicitud inválida: {_validation_message(exc)}", file=sys.stderr)
        return 2
    except SimulationError as exc:
        print(f"No se puede simular: {exc}", file=sys.stderr)
        return 2
    print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
