"""Verifica que el CSV de Kaggle esté en su lugar y sea la versión esperada.

No descarga nada automáticamente: Kaggle exige cuenta y aceptar la licencia
(CC BY-NC-SA 4.0), y el proyecto no maneja credenciales.

Pasos manuales:
1. Entra a https://www.kaggle.com/datasets/joniarroba/noshowappointments
2. Descarga el ZIP y extrae ``KaggleV2-May-2016.csv``.
3. Guárdalo como ``data/raw/data.csv``.
4. Ejecuta ``python scripts/download_data.py`` para verificar el hash.
"""

from __future__ import annotations

import sys

from noshow_guard.config import PATHS, RAW_SHA256
from noshow_guard.data import file_sha256


def main() -> int:
    """Devuelve 0 si el CSV existe y su hash coincide; 1 si falta o es distinto."""
    if not PATHS.raw_csv.exists():
        print(__doc__)
        print(f"Falta el archivo: {PATHS.raw_csv}", file=sys.stderr)
        return 1

    actual = file_sha256(PATHS.raw_csv)
    if actual != RAW_SHA256:
        print(f"Hash distinto.\n  esperado: {RAW_SHA256}\n  actual:   {actual}", file=sys.stderr)
        return 1

    print(f"OK: {PATHS.raw_csv} (sha256 {actual[:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
