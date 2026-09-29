"""Explicaciones SHAP: importancia global y factores principales de un caso.

Los valores SHAP salen de ``NoShowModel.contributions``: son exactos para la regresión logística
(coeficiente por desviación respecto a la media de train, sumado por feature original) y están
en log-odds del modelo **sin calibrar**. La calibración es monótona, así que el signo de cada
factor (sube o baja el riesgo) se conserva, pero la magnitud no se traduce 1:1 a puntos de
probabilidad calibrada. Además, SHAP describe al modelo, no causas: una contribución alta de la
antelación no prueba que acortar la antelación reduzca el no-show.
"""

from __future__ import annotations

import pandas as pd

from noshow_guard.features import FEATURE_COLUMNS


def global_importance(contributions: pd.DataFrame) -> pd.Series:
    """Media del valor absoluto SHAP por feature, de mayor a menor."""
    return contributions[list(FEATURE_COLUMNS)].abs().mean().sort_values(ascending=False)


def top_factors(
    contribution_row: pd.Series, feature_row: pd.Series, k: int = 3
) -> list[dict[str, object]]:
    """Los ``k`` features con mayor |SHAP| en un caso, con el signo de su efecto y su valor."""
    shap_values = contribution_row[list(FEATURE_COLUMNS)].astype(float)
    order = shap_values.abs().sort_values(ascending=False, kind="stable").index[:k]
    return [
        {
            "feature": name,
            "effect": "+" if shap_values[name] > 0 else "-",
            "value": _plain(feature_row[name]),
            "shap": round(float(shap_values[name]), 4),
        }
        for name in order
    ]


def _plain(value: object) -> object:
    """Convierte escalares numpy a tipos nativos para serializar en JSON."""
    return value.item() if hasattr(value, "item") else value
