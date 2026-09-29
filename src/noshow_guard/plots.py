"""Figuras de evaluación (se guardan en reports/figures)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from noshow_guard.evaluation import Thresholds


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pr_curves(y: np.ndarray, scores: dict[str, np.ndarray], path: Path) -> None:
    """Curvas precisión-recall de varios modelos sobre el mismo conjunto."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, s in scores.items():
        precision, recall, _ = precision_recall_curve(y, s)
        ax.plot(recall, precision, label=name)
    ax.axhline(np.mean(y), color="gray", ls="--", lw=1, label=f"prevalencia {np.mean(y):.3f}")
    ax.set(xlabel="Recall", ylabel="Precisión", title="Curvas PR (test)", ylim=(0, 1))
    ax.legend(fontsize=8)
    _save(fig, path)


def calibration_plot(y: np.ndarray, probs: dict[str, np.ndarray], path: Path) -> None:
    """Diagrama de confiabilidad (bins por cuantiles) e histograma de probabilidades."""
    fig, (ax, hx) = plt.subplots(2, 1, figsize=(6, 6), height_ratios=[3, 1], sharex=True)
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
    for name, p in probs.items():
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", label=name)
        hx.hist(p, bins=40, range=(0, 1), alpha=0.5, label=name)
    ax.set(ylabel="Fracción observada de no-show", title="Calibración (test)")
    ax.legend(fontsize=8)
    hx.set(xlabel="Probabilidad predicha", ylabel="Citas")
    _save(fig, path)


def cost_curve(
    grid: pd.DataFrame, chosen: Thresholds, references: dict[str, float], path: Path
) -> None:
    """Costo medio vs umbral estándar, con el umbral reforzado fijo en el elegido."""
    line = grid[np.isclose(grid["reinforced"], chosen.reinforced)].sort_values("standard")
    line = line[line["standard"] <= chosen.reinforced]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(line["standard"], line["cost"], label=f"política (reforzado ≥ {chosen.reinforced:.2f})")
    for i, (name, value) in enumerate(references.items()):
        ax.axhline(value, ls="--", lw=1, color=f"C{i + 1}", label=name)
    ax.axvline(
        chosen.standard, color="k", lw=1, ls=":", label=f"umbral estándar {chosen.standard:.2f}"
    )
    ax.set(
        xlabel="Umbral de recordatorio estándar (prob. calibrada)",
        ylabel="Costo medio por cita (unidades)",
        title="Costo esperado vs umbral (validación)",
    )
    ax.legend(fontsize=8)
    _save(fig, path)


def shap_summary(contributions: pd.DataFrame, features: pd.DataFrame, path_prefix: Path) -> None:
    """Gráficos SHAP globales: barras de |SHAP| medio y beeswarm.

    ``features`` debe ser numérico (el barrio ya codificado) para colorear el beeswarm.
    """
    import shap

    values = contributions[features.columns].to_numpy()
    shap.summary_plot(values, features, plot_type="bar", show=False)
    _save(plt.gcf(), path_prefix.with_name(path_prefix.name + "_bar.png"))
    shap.summary_plot(values, features, show=False)
    _save(plt.gcf(), path_prefix.with_name(path_prefix.name + "_beeswarm.png"))
