"""Entrenamiento de humo de punta a punta con datos sintéticos (para CI).

Recorre datos -> features -> split -> entrenamiento -> guardado -> simulación -> registro
en un directorio temporal, sin el CSV de Kaggle y sin tocar ``models/`` ni ``reports/``.

Uso: ``python -m noshow_guard.smoke`` (o ``make smoke``).
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date, time
from pathlib import Path

from noshow_guard.config import Paths, SearchConfig
from noshow_guard.data import validate
from noshow_guard.features import build_feature_frame, temporal_split
from noshow_guard.schemas import SimulationRequest
from noshow_guard.simulator import Simulator
from noshow_guard.synthetic import synthetic_clean
from noshow_guard.train import fit_all, save

SMOKE_SEARCH = SearchConfig(
    n_iter=2, max_estimators=60, early_stopping_rounds=10, calibration_folds=3
)


def run(root: Path) -> dict[str, object]:
    """Ejecuta el pipeline completo bajo ``root`` y devuelve un resumen verificable."""
    paths = Paths(root=root)
    clean = validate(synthetic_clean())
    paths.clean_parquet.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(paths.clean_parquet, index=False)

    splits = temporal_split(build_feature_frame(clean, clean))
    version = save(fit_all(splits, search=SMOKE_SEARCH), splits, paths)

    simulator = Simulator.from_disk(paths)
    requests = [
        SimulationRequest(
            patient_id=str(clean["patient_id"].iloc[0]),
            appointment_date=date(2016, 6, 10),
            appointment_time=time(9, 0),
            as_of=date(2016, 6, 1),
        ),
        SimulationRequest.model_validate(
            {
                "new_patient": {"age": 30, "gender": "F", "neighbourhood": "centro"},
                "appointment_date": "2016-06-10",
                "appointment_time": "10:00",
                "as_of": "2016-06-01",
            }
        ),
    ]
    results = [simulator.simulate(r) for r in requests]
    logged = simulator.registry.read() if simulator.registry else None
    return {
        "model_version": version,
        "statuses": [r.status for r in results],
        "probabilities": [r.probability_no_show for r in results],
        "logged_rows": 0 if logged is None else len(logged),
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="noshow_smoke_") as tmp:
        summary = run(Path(tmp))
    print(summary)
    ok = summary["statuses"] == ["ok", "ok"] and summary["logged_rows"] == 2
    probabilities = summary["probabilities"]
    ok = ok and isinstance(probabilities, list) and all(0 <= p <= 1 for p in probabilities)
    if not ok:
        print("Smoke FALLÓ", file=sys.stderr)
        return 1
    print("Smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
