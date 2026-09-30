"""Monitoreo de drift con implementación propia (numpy + scipy).

- PSI por variable: bins por cuantiles de la referencia (numéricas) o por categoría.
- KS para numéricas y chi-cuadrado para categóricas (p-valores informativos).
- Misma comparación para la probabilidad predicha por el modelo.

El veredicto se basa en el PSI y no en los p-valores: con miles de filas casi cualquier
diferencia resulta "significativa" aunque no importe.

Uso: ``python -m noshow_guard.drift report`` (o ``make monitor``).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from noshow_guard.config import DRIFT, PATHS, SEED, DriftConfig, Paths
from noshow_guard.features import FEATURE_COLUMNS
from noshow_guard.formatting import es_number
from noshow_guard.model import NoShowModel, load_model
from noshow_guard.registry import Registry

NUMERIC: tuple[str, ...] = (
    "lead_time_days",
    "age",
    "prev_appointments",
    "prev_no_shows",
    "prev_no_show_rate",
    "probability",
)
CATEGORICAL: tuple[str, ...] = tuple(c for c in FEATURE_COLUMNS if c not in NUMERIC)
Status = Literal["estable", "vigilar", "alerta"]
Verdict = Literal["estable", "vigilar", "reentrenar", "muestra_insuficiente"]
_EPS = 1e-4


# --- Métricas ------------------------------------------------------------------


def _psi(ref_share: np.ndarray, cur_share: np.ndarray) -> float:
    """PSI = sum((cur - ref) * ln(cur / ref)), con piso para bins vacíos."""
    ref = np.clip(ref_share, _EPS, None)
    cur = np.clip(cur_share, _EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def quantile_edges(reference: np.ndarray, bins: int = DRIFT.bins) -> np.ndarray:
    """Cortes internos por cuantiles de la referencia, sin repetidos."""
    return np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))


def psi_numeric(reference: np.ndarray, current: np.ndarray, bins: int = DRIFT.bins) -> float:
    """PSI con bins por cuantiles de la referencia.

    Los bins son cerrados a la derecha, ``(-inf, e1], (e1, e2], ..., (ek, inf)``, para que
    un valor muy repetido (p. ej. 0 en ``prev_no_shows``, 95 % de train) tenga su propio bin
    y no se mezcle con los valores mayores.
    """
    inner = quantile_edges(reference, bins)
    n_bins = len(inner) + 1
    ref_counts = np.bincount(np.searchsorted(inner, reference, side="left"), minlength=n_bins)
    cur_counts = np.bincount(np.searchsorted(inner, current, side="left"), minlength=n_bins)
    return _psi(ref_counts / len(reference), cur_counts / len(current))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """PSI por categoría; una categoría nueva cuenta como bin vacío en la referencia."""
    ref = reference.value_counts(normalize=True)
    cur = current.value_counts(normalize=True)
    categories = ref.index.union(cur.index)
    return _psi(
        ref.reindex(categories, fill_value=0).to_numpy(),
        cur.reindex(categories, fill_value=0).to_numpy(),
    )


def ks_pvalue(reference: np.ndarray, current: np.ndarray) -> float:
    return float(stats.ks_2samp(reference, current).pvalue)


def chi2_pvalue(reference: pd.Series, current: pd.Series) -> float:
    """Chi-cuadrado sobre la tabla de conteos referencia vs actual."""
    table = pd.concat(
        [reference.value_counts(), current.value_counts()], axis=1, keys=["ref", "cur"]
    ).fillna(0)
    if len(table) < 2:
        return 1.0
    return float(stats.chi2_contingency(table.T.to_numpy()).pvalue)


def psi_status(psi: float, cfg: DriftConfig = DRIFT) -> Status:
    if psi > cfg.psi_alert:
        return "alerta"
    if psi >= cfg.psi_watch:
        return "vigilar"
    return "estable"


# --- Comparación ---------------------------------------------------------------


def _summary(values: pd.Series, numeric: bool) -> str:
    """Resumen legible: media para numéricas, categoría más frecuente y su peso para el resto."""
    if numeric:
        return f"media {es_number(values.mean())}"
    top = values.value_counts(normalize=True)
    return f"{top.index[0]} ({es_number(100 * top.iloc[0], 1)} %)"


def compare(
    reference: pd.DataFrame, current: pd.DataFrame, cfg: DriftConfig = DRIFT
) -> pd.DataFrame:
    """Una fila por variable: PSI, estado, prueba estadística y resúmenes."""
    rows = []
    for column in [*NUMERIC, *CATEGORICAL]:
        if column not in reference or column not in current:
            continue
        numeric = column in NUMERIC
        ref, cur = reference[column], current[column]
        if numeric:
            psi = psi_numeric(ref.to_numpy(float), cur.to_numpy(float), cfg.bins)
            test, pvalue = "KS", ks_pvalue(ref.to_numpy(float), cur.to_numpy(float))
        else:
            psi = psi_categorical(ref.astype(str), cur.astype(str))
            test, pvalue = "chi²", chi2_pvalue(ref.astype(str), cur.astype(str))
        rows.append(
            {
                "variable": column,
                "psi": psi,
                "estado": psi_status(psi, cfg),
                "prueba": test,
                "p_valor": pvalue,
                "referencia": _summary(ref, numeric),
                "actual": _summary(cur, numeric),
            }
        )
    return pd.DataFrame(rows)


def verdict(table: pd.DataFrame, n: int, cfg: DriftConfig = DRIFT) -> Verdict:
    """Estable, vigilar o reentrenar (alerta en una variable clave); nada si la muestra es chica."""
    if n < cfg.min_sample:
        return "muestra_insuficiente"
    key_alert = table["estado"].eq("alerta") & table["variable"].isin(cfg.key_variables)
    if key_alert.any():
        return "reentrenar"
    if table["estado"].isin(["vigilar", "alerta"]).any():
        return "vigilar"
    return "estable"


# --- Lotes "de producción" -----------------------------------------------------


def control_batch(reference: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Remuestreo de la referencia: misma distribución, debe salir estable."""
    return reference.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)


def induced_drift_batch(reference: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Drift inducido a propósito sobre un remuestreo de la referencia.

    Antelación x3, edad a la mitad (población más joven) y todos los pacientes nuevos.
    """
    batch = control_batch(reference, n, seed + 1)
    return batch.assign(
        lead_time_days=batch["lead_time_days"] * 3,
        age=(batch["age"] * 0.5).astype(int),
        has_history=0,
        prev_appointments=0,
        prev_no_shows=0,
        prev_no_show_rate=0.0,
    )


def registry_features(simulations: pd.DataFrame) -> pd.DataFrame:
    """Features de las simulaciones registradas con resultado (status ok)."""
    if simulations.empty:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))
    ok = simulations[simulations["status"] == "ok"]
    records = [json.loads(raw) for raw in ok["features_json"]]
    return pd.DataFrame(records, columns=list(FEATURE_COLUMNS))


def with_probability(model: NoShowModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Agrega la probabilidad calibrada del modelo a un lote de features."""
    if frame.empty:
        return frame.assign(probability=pd.Series(dtype=float))
    return frame[list(FEATURE_COLUMNS)].assign(probability=model.predict_proba(frame))


# --- Reporte -------------------------------------------------------------------


@dataclass
class BatchResult:
    name: str
    description: str
    n: int
    table: pd.DataFrame
    verdict: Verdict


def evaluate_batches(
    reference: pd.DataFrame,
    batches: dict[str, tuple[str, pd.DataFrame]],
    cfg: DriftConfig = DRIFT,
) -> list[BatchResult]:
    """Compara cada lote contra la referencia (ambos con columna ``probability``)."""
    results = []
    for name, (description, frame) in batches.items():
        table = compare(reference, frame, cfg) if len(frame) else pd.DataFrame()
        results.append(
            BatchResult(
                name,
                description,
                len(frame),
                table,
                verdict(table, len(frame), cfg) if len(frame) else "muestra_insuficiente",
            )
        )
    return results


def _fmt_p(p: float) -> str:
    return "< 0,001" if p < 0.001 else es_number(p)


def render_markdown(
    results: list[BatchResult], model_version: str, n_reference: int, cfg: DriftConfig = DRIFT
) -> str:
    """Reporte en Markdown con resumen, tablas por lote y figuras."""
    lines = [
        "# Reporte de drift",
        "",
        f"- Generado: {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
        f"- Modelo: `{model_version}`",
        f"- Referencia: features de **train** ({es_number(n_reference, 0)} citas) y su "
        "probabilidad predicha",
        f"- PSI < {es_number(cfg.psi_watch, 2)} estable · {es_number(cfg.psi_watch, 2)} a "
        f"{es_number(cfg.psi_alert, 2)} vigilar · > {es_number(cfg.psi_alert, 2)} alerta",
        f"- Veredicto «reentrenar»: alerta en una variable clave ({', '.join(cfg.key_variables)}). "
        "Es una señal para evaluar un reentrenamiento, no un reentrenamiento automático "
        "(ver `docs/retraining_plan.md`).",
        f"- Con menos de {cfg.min_sample} filas no se emite veredicto.",
        "",
        "## Resumen",
        "",
        "| Lote | Filas | Veredicto | Variables en alerta | Variables a vigilar |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        alert = watch = "-"
        if not r.table.empty and r.verdict != "muestra_insuficiente":
            alert = ", ".join(r.table.loc[r.table["estado"] == "alerta", "variable"]) or "-"
            watch = ", ".join(r.table.loc[r.table["estado"] == "vigilar", "variable"]) or "-"
        lines.append(f"| {r.name} | {es_number(r.n, 0)} | **{r.verdict}** | {alert} | {watch} |")
    lines += [
        "",
        "![PSI por variable y lote](figures/drift_psi.png)",
        "",
        "![Probabilidad predicha](figures/drift_probability.png)",
        "",
        "Los p-valores (KS y chi²) se muestran como referencia. Con miles de filas casi "
        "cualquier diferencia es estadísticamente significativa aunque sea irrelevante, por eso "
        "el veredicto usa el PSI.",
    ]
    for r in results:
        lines += ["", f"## {r.name}", "", r.description, ""]
        if r.verdict == "muestra_insuficiente":
            lines.append(f"Solo {r.n} fila(s): muestra insuficiente para evaluar drift.")
            continue
        lines += [
            "| Variable | PSI | Estado | Prueba | p-valor | Referencia | Actual |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in r.table.to_dict("records"):
            lines.append(
                f"| {row['variable']} | {es_number(float(row['psi']))} | {row['estado']} | "
                f"{row['prueba']} | {_fmt_p(float(row['p_valor']))} | {row['referencia']} | "
                f"{row['actual']} |"
            )
    return "\n".join(lines) + "\n"


def plot_report(
    results: list[BatchResult],
    reference: pd.DataFrame,
    batches: dict[str, pd.DataFrame],
    paths: Paths,
    cfg: DriftConfig = DRIFT,
) -> None:
    """Heatmap de PSI (variables x lotes) e histogramas de probabilidad predicha."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evaluated = [r for r in results if r.verdict != "muestra_insuficiente"]
    paths.figures.mkdir(parents=True, exist_ok=True)
    if evaluated:
        psi = pd.DataFrame({r.name: r.table.set_index("variable")["psi"] for r in evaluated})
        fig, ax = plt.subplots(figsize=(1.8 * len(evaluated) + 3, 6))
        im = ax.imshow(np.minimum(psi.to_numpy(), 1.0), cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(psi.shape[1]), psi.columns, rotation=20, ha="right")
        ax.set_yticks(range(psi.shape[0]), psi.index)
        for (i, j), value in np.ndenumerate(psi.to_numpy()):
            color = "white" if value > 0.6 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)
        ax.set_title("PSI por variable (color saturado en 1,0)")
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(paths.figures / "drift_psi.png", dpi=110)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(reference["probability"], bins=40, range=(0, 1), density=True, alpha=0.5,
            label="referencia (train)")  # fmt: skip
    for name, frame in batches.items():
        if len(frame) >= cfg.min_sample:
            ax.hist(frame["probability"], bins=40, range=(0, 1), density=True,
                    histtype="step", lw=1.5, label=name)  # fmt: skip
    ax.set(xlabel="Probabilidad calibrada de no-show", ylabel="Densidad",
           title="Distribución de la probabilidad predicha")  # fmt: skip
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.figures / "drift_probability.png", dpi=110)
    plt.close(fig)


def build_report(
    model: NoShowModel,
    model_version: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    simulations: pd.DataFrame,
    cfg: DriftConfig = DRIFT,
    paths: Paths = PATHS,
) -> list[BatchResult]:
    """Arma los lotes, calcula el drift y escribe el reporte y las figuras."""
    reference = with_probability(model, train)
    size = min(cfg.batch_size, len(reference))
    batches = {
        "simulaciones registradas": (
            "Simulaciones guardadas en `data/simulations.db` (solo las que tienen probabilidad).",
            with_probability(model, registry_features(simulations)),
        ),
        "control sin drift": (
            f"Remuestreo con reemplazo de {es_number(size, 0)} citas de train. Debe salir estable.",
            with_probability(model, control_batch(train, size)),
        ),
        "drift inducido": (
            f"Remuestreo de {es_number(size, 0)} citas de train con drift a propósito: "
            "antelación triplicada, edad a la mitad (población más joven) y todos los "
            "pacientes nuevos (sin historial).",
            with_probability(model, induced_drift_batch(train, size)),
        ),
        "test real (junio 2016)": (
            "Citas reales de test (2016-06-01 a 2016-06-08): el drift que de verdad ocurrió "
            "después del periodo de entrenamiento.",
            with_probability(model, test),
        ),
    }
    results = evaluate_batches(reference, batches, cfg)
    paths.reports.mkdir(parents=True, exist_ok=True)
    paths.drift_report.write_text(
        render_markdown(results, model_version, len(reference), cfg), encoding="utf-8"
    )
    plot_report(results, reference, {k: v for k, (_, v) in batches.items()}, paths, cfg)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="noshow_guard.drift", description="Monitoreo de drift.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("report", help="Genera reports/drift_report.md.")
    parser.parse_args(argv)

    model, metadata = load_model()
    train = pd.read_parquet(PATHS.features_dir / "train.parquet")
    test = pd.read_parquet(PATHS.features_dir / "test.parquet")
    simulations = Registry(PATHS.registry_db).read()
    results = build_report(model, metadata["model_version"], train, test, simulations)
    for r in results:
        print(f"{r.name:<28} n={r.n:>6}  veredicto={r.verdict}")
    print(f"Reporte: {PATHS.drift_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
