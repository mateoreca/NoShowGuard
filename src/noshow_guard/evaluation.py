"""Métricas, política de acciones por costo y análisis por segmentos (funciones puras)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from noshow_guard.config import COSTS, RISK, CostConfig, RiskConfig

ACTIONS: tuple[str, ...] = (
    "sin_accion",
    "recordatorio_estandar",
    "recordatorio_reforzado_con_confirmacion",
)
THRESHOLD_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)


@dataclass(frozen=True)
class Thresholds:
    """Umbrales de la política de 3 acciones sobre la probabilidad calibrada p.

    p < standard: sin acción; standard <= p < reinforced: estándar; p >= reinforced: reforzado.
    """

    standard: float
    reinforced: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


# --- Métricas ------------------------------------------------------------------


def score_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Métricas de ranking y de probabilidad. PR-AUC = average precision."""
    out = {"pr_auc": float(average_precision_score(y, p)), "prevalence": float(np.mean(y))}
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
    if np.all((p >= 0) & (p <= 1)):
        out["brier"] = float(brier_score_loss(y, p))
        out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
    return out


def binary_metrics(y: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    """Precisión, recall (TPR), FPR y tasa de marcados para una decisión binaria."""
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    tp = int(np.sum(flagged & y))
    fp = int(np.sum(flagged & ~y))
    return {
        "precision": tp / flagged.sum() if flagged.sum() else float("nan"),
        "recall": tp / y.sum() if y.sum() else float("nan"),
        "fpr": fp / (~y).sum() if (~y).sum() else float("nan"),
        "flagged_rate": float(flagged.mean()),
    }


# --- Política por costo --------------------------------------------------------


def analytic_thresholds(costs: CostConfig = COSTS) -> Thresholds:
    """Umbrales óptimos si las probabilidades están bien calibradas.

    Costo esperado de una acción con probabilidad p: ``costo + no_show_cost * p * (1 - efecto)``.
    Estándar gana a nada si ``p > c_s / (C * e_s)``; reforzado gana a estándar si
    ``p > (c_r - c_s) / (C * (e_r - e_s))``.
    """
    standard = costs.standard_cost / (costs.no_show_cost * costs.standard_effect)
    reinforced = (costs.reinforced_cost - costs.standard_cost) / (
        costs.no_show_cost * (costs.reinforced_effect - costs.standard_effect)
    )
    # Si reforzado domina antes que estándar, el tramo estándar desaparece.
    if reinforced <= standard:
        standard = reinforced = costs.reinforced_cost / (
            costs.no_show_cost * costs.reinforced_effect
        )
    return Thresholds(standard=min(standard, 1.0), reinforced=min(reinforced, 1.0))


def assign_actions(p: np.ndarray, thresholds: Thresholds) -> np.ndarray:
    """Índice de acción por cita: 0 sin acción, 1 estándar, 2 reforzado."""
    p = np.asarray(p)
    return np.where(p >= thresholds.reinforced, 2, np.where(p >= thresholds.standard, 1, 0))


def action_costs(actions: np.ndarray, y: np.ndarray, costs: CostConfig = COSTS) -> np.ndarray:
    """Costo por cita dado el resultado observado y el efecto SUPUESTO de la acción."""
    action_cost = np.array([0.0, costs.standard_cost, costs.reinforced_cost])[actions]
    effect = np.array([0.0, costs.standard_effect, costs.reinforced_effect])[actions]
    return action_cost + costs.no_show_cost * np.asarray(y) * (1.0 - effect)


def mean_cost(
    p: np.ndarray, y: np.ndarray, thresholds: Thresholds, costs: CostConfig = COSTS
) -> float:
    """Costo medio por cita de la política de umbrales."""
    return float(action_costs(assign_actions(p, thresholds), y, costs).mean())


def cost_grid(
    p: np.ndarray, y: np.ndarray, costs: CostConfig = COSTS, grid: np.ndarray = THRESHOLD_GRID
) -> pd.DataFrame:
    """Costo medio para cada par (standard, reinforced) con reinforced >= standard."""
    rows = [
        {"standard": t1, "reinforced": t2, "cost": mean_cost(p, y, Thresholds(t1, t2), costs)}
        for t1 in grid
        for t2 in grid
        if t2 >= t1
    ]
    return pd.DataFrame(rows)


def optimize_thresholds(
    p: np.ndarray, y: np.ndarray, costs: CostConfig = COSTS, grid: np.ndarray = THRESHOLD_GRID
) -> tuple[Thresholds, pd.DataFrame]:
    """Umbrales que minimizan el costo medio observado.

    Muchos pares empatan (por ejemplo, cualquier umbral por encima de la mayor probabilidad);
    el empate se resuelve con el par más cercano a los umbrales analíticos.
    """
    table = cost_grid(p, y, costs, grid)
    best = table[table["cost"] <= table["cost"].min() + 1e-12]
    ref = analytic_thresholds(costs)
    distance = (best["standard"] - ref.standard).abs() + (best["reinforced"] - ref.reinforced).abs()
    row = best.loc[distance.idxmin()]
    return Thresholds(float(row["standard"]), float(row["reinforced"])), table


# --- Nivel de riesgo -----------------------------------------------------------


def risk_cutoffs(p_reference: np.ndarray, risk: RiskConfig = RISK) -> dict[str, float]:
    """Cortes de nivel de riesgo a partir de percentiles de una distribución de referencia."""
    return {
        "medio": float(np.quantile(p_reference, risk.medium_quantile)),
        "alto": float(np.quantile(p_reference, risk.high_quantile)),
    }


def risk_level(p: float, cutoffs: dict[str, float]) -> str:
    """ "bajo", "medio" o "alto" según los cortes (inclusivos por abajo)."""
    if p >= cutoffs["alto"]:
        return "alto"
    if p >= cutoffs["medio"]:
        return "medio"
    return "bajo"


# --- Segmentos -----------------------------------------------------------------


def segment_table(
    y: np.ndarray, p: np.ndarray, groups: pd.Series, threshold: float
) -> pd.DataFrame:
    """Métricas por grupo: tamaño, prevalencia, prob. media, ranking y decisión binaria."""
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "g": groups.to_numpy()})
    rows = []
    for name, part in frame.groupby("g", observed=True, sort=True):
        metrics = score_metrics(part["y"].to_numpy(), part["p"].to_numpy())
        rows.append(
            {
                "group": str(name),
                "n": len(part),
                "prevalence": metrics["prevalence"],
                "mean_p": float(part["p"].mean()),
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics.get("roc_auc", float("nan")),
                "brier": metrics["brier"],
                **binary_metrics(part["y"].to_numpy(), (part["p"] >= threshold).to_numpy()),
            }
        )
    return pd.DataFrame(rows)
