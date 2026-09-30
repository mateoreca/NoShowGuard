"""Simulación de impacto de políticas de recordatorio. ES UNA SIMULACIÓN, NO UN RESULTADO REAL.

Sobre las citas de test (resultado observado ``y`` y probabilidad calibrada del modelo) se
comparan tres políticas bajo supuestos explícitos del efecto de cada recordatorio:

- no hacer nada,
- recordatorio estándar a todos,
- según el modelo (umbrales por costo guardados en ``metadata.json``).

Para una cita con resultado ``y`` y una acción con efecto supuesto ``e``, los no-shows
evitados esperados son ``y * e`` y el costo es ``costo_acción + costo_hueco * y * (1 - e)``.
Los umbrales del modelo quedan fijos (los desplegados); solo cambian los supuestos.

Uso: ``python -m noshow_guard.impact`` (o ``make impact``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from noshow_guard.actions import thresholds_from_metadata
from noshow_guard.config import IMPACT, PATHS, CostConfig, ImpactConfig, Paths
from noshow_guard.evaluation import Thresholds, action_costs, assign_actions
from noshow_guard.formatting import es_number
from noshow_guard.model import load_model

POLICIES: tuple[str, ...] = ("no_hacer_nada", "recordar_a_todos", "segun_modelo")
BANNER = (
    "> **SIMULACIÓN, NO RESULTADO REAL.** Los efectos de los recordatorios son supuestos "
    "configurables (`ImpactConfig` en `config.py`); no se estimaron con datos. Las cifras "
    "dicen qué pasaría *si* los supuestos fueran ciertos."
)


def policy_actions(p: np.ndarray, thresholds: Thresholds) -> dict[str, np.ndarray]:
    """Acción por cita (0 nada, 1 estándar, 2 reforzado) para cada política."""
    n = len(p)
    return {
        "no_hacer_nada": np.zeros(n, dtype=int),
        "recordar_a_todos": np.ones(n, dtype=int),
        "segun_modelo": assign_actions(p, thresholds),
    }


def evaluate_policy(actions: np.ndarray, y: np.ndarray, costs: CostConfig) -> dict[str, float]:
    """No-shows evitados (esperados), recordatorios y costo total de una política."""
    effect = np.array([0.0, costs.standard_effect, costs.reinforced_effect])[actions]
    return {
        "no_shows_sin_accion": float(np.sum(y)),
        "no_shows_evitados": float(np.sum(y * effect)),
        "recordatorios_estandar": int(np.sum(actions == 1)),
        "recordatorios_reforzados": int(np.sum(actions == 2)),
        "costo_total": float(action_costs(actions, y, costs).sum()),
    }


def impact_table(
    p: np.ndarray,
    y: np.ndarray,
    thresholds: Thresholds,
    base_costs: CostConfig,
    impact: ImpactConfig = IMPACT,
) -> pd.DataFrame:
    """Una fila por escenario y política, con el ahorro frente a no hacer nada."""
    rows = []
    for scenario in impact.scenarios:
        costs = replace(
            base_costs,
            standard_effect=scenario.standard_effect,
            reinforced_effect=scenario.reinforced_effect,
        )
        results = {
            name: evaluate_policy(a, y, costs) for name, a in policy_actions(p, thresholds).items()
        }
        baseline = results["no_hacer_nada"]["costo_total"]
        for name in POLICIES:
            r = results[name]
            reminders = r["recordatorios_estandar"] + r["recordatorios_reforzados"]
            rows.append(
                {
                    "escenario": scenario.name,
                    "efecto_estandar": scenario.standard_effect,
                    "efecto_reforzado": scenario.reinforced_effect,
                    "politica": name,
                    **r,
                    "recordatorios": reminders,
                    "evitados_por_100_recordatorios": (
                        100 * r["no_shows_evitados"] / reminders if reminders else 0.0
                    ),
                    "ahorro_vs_nada": baseline - r["costo_total"],
                    "ahorro_por_cita": (baseline - r["costo_total"]) / len(y),
                }
            )
    return pd.DataFrame(rows)


def render_markdown(
    table: pd.DataFrame, thresholds: Thresholds, costs: CostConfig, n: int, model_version: str
) -> str:
    """Reporte en Markdown con los supuestos visibles antes de cualquier cifra."""
    lines = [
        "# Simulación de impacto de recordatorios",
        "",
        BANNER,
        "",
        f"- Citas: test real, {es_number(n, 0)} citas del 2016-06-01 al 2016-06-08, con su "
        "resultado observado.",
        f"- Modelo: `{model_version}`. Umbrales fijos (elegidos en validación con el escenario "
        f"base): recordatorio estándar si p ≥ {es_number(thresholds.standard, 2)}, reforzado "
        f"si p ≥ {es_number(thresholds.reinforced, 2)}.",
        f"- Costos supuestos (unidades, 1 = un recordatorio estándar): hueco vacío "
        f"{es_number(costs.no_show_cost, 0)}, estándar {es_number(costs.standard_cost, 0)}, "
        f"reforzado {es_number(costs.reinforced_cost, 0)}.",
        "",
        "## Supuestos de efecto (reducción relativa del no-show)",
        "",
        "| Escenario | Recordatorio estándar | Reforzado con confirmación |",
        "|---|---|---|",
    ]
    for s in table.drop_duplicates("escenario").to_dict("records"):
        lines.append(
            f"| {s['escenario']} | {es_number(100 * float(s['efecto_estandar']), 0)} % | "
            f"{es_number(100 * float(s['efecto_reforzado']), 0)} % |"
        )
    lines += [
        "",
        "Como referencia no causal: en el EDA, dentro de cada tramo de antelación, las citas con "
        "SMS tuvieron entre 10 % y 25 % menos no-show relativo que las sin SMS. El SMS no se "
        "asignó al azar, así que eso no mide su efecto.",
        "",
        "## Resultados",
        "",
        "| Escenario | Política | Recordatorios | No-shows evitados (esperados) | "
        "Evitados por 100 recordatorios | Costo total | Ahorro vs no hacer nada |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in table.to_dict("records"):
        lines.append(
            f"| {r['escenario']} | {r['politica']} | {es_number(float(r['recordatorios']), 0)} | "
            f"{es_number(float(r['no_shows_evitados']), 1)} de "
            f"{es_number(float(r['no_shows_sin_accion']), 0)} | "
            f"{es_number(float(r['evitados_por_100_recordatorios']), 1)} | "
            f"{es_number(float(r['costo_total']), 0)} | "
            f"{es_number(float(r['ahorro_vs_nada']), 0)} |"
        )
    lines += ["", "![Impacto simulado](figures/impact.png)", ""]
    return "\n".join(lines)


def plot_impact(table: pd.DataFrame, paths: Paths) -> None:
    """Ahorro vs no hacer nada por política y escenario."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = table.pivot(index="escenario", columns="politica", values="ahorro_vs_nada")
    pivot = pivot.loc[[s.name for s in IMPACT.scenarios if s.name in pivot.index], list(POLICIES)]
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot.plot(kind="bar", ax=ax, rot=0)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        xlabel="Escenario de supuestos",
        ylabel="Ahorro vs no hacer nada (unidades)",
        title="SIMULACIÓN: ahorro por política (test, supuestos explícitos)",
    )
    ax.legend(fontsize=8)
    paths.figures.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(paths.figures / "impact.png", dpi=110)
    plt.close(fig)


def main(paths: Paths = PATHS) -> int:
    model, metadata = load_model(paths.model_metadata)
    test = pd.read_parquet(paths.features_dir / "test.parquet")
    p = model.predict_proba(test)
    y = test["no_show"].to_numpy()
    thresholds = thresholds_from_metadata(metadata)
    costs = CostConfig(**metadata["costs"])

    table = impact_table(p, y, thresholds, costs)
    paths.reports.mkdir(parents=True, exist_ok=True)
    paths.impact_report.write_text(
        render_markdown(table, thresholds, costs, len(y), metadata["model_version"]),
        encoding="utf-8",
    )
    (paths.reports / "impact.json").write_text(
        json.dumps(
            {
                "es_simulacion": True,
                "supuestos": [asdict(s) for s in IMPACT.scenarios],
                "costos": asdict(costs),
                "umbrales": thresholds.to_dict(),
                "resultados": table.to_dict("records"),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_impact(table, paths)
    columns = ["escenario", "politica", "recordatorios", "no_shows_evitados", "ahorro_vs_nada"]
    print("SIMULACIÓN (supuestos explícitos, no resultados reales)")
    print(table[columns].round(1).to_string(index=False))
    print(f"Reporte: {paths.impact_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
