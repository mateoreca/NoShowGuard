"""Formato de números para los reportes en español."""

from __future__ import annotations


def es_number(value: float, decimals: int = 3) -> str:
    """Punto de miles y coma decimal: ``es_number(12345.678, 1) == '12.345,7'``."""
    text = f"{float(value):,.{decimals}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")
