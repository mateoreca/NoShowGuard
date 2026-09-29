"""Pipelines de modelos, calibradores y el artefacto final ``NoShowModel``.

Modelo principal: regresión logística calibrada (ver docs/adr/0001-modelo-principal.md).
LightGBM se entrena solo como comparación y no se guarda como artefacto.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler, TargetEncoder

from noshow_guard.config import PATHS, SEED
from noshow_guard.features import FEATURE_COLUMNS

CalibrationMethod = Literal["isotonic", "platt"]
_EPS = 1e-6

# Features binarias: entran sin transformar al modelo lineal.
_PASSTHROUGH = ("is_male", "scholarship", "hypertension", "diabetes", "alcoholism", "has_history")
_SCALED = ("handicap", "prev_appointments", "prev_no_shows", "prev_no_show_rate")


def _neighbourhood_encoder() -> TargetEncoder:
    """Target encoding suavizado; en ``fit_transform`` usa validación cruzada interna."""
    return TargetEncoder(target_type="binary", smooth="auto", cv=5, shuffle=True, random_state=SEED)


def _quantile_bins() -> KBinsDiscretizer:
    return KBinsDiscretizer(
        n_bins=10,
        encode="onehot-dense",
        strategy="quantile",
        quantile_method="averaged_inverted_cdf",
    )


def make_logistic_pipeline() -> Pipeline:
    """Regresión logística con un transformador por feature original.

    Cada transformador se llama como su feature, así las columnas de salida
    (``<feature>__<columna>``) se agregan sin ambigüedad al explicar con SHAP.
    Edad y antelación van en bins por cuantiles porque su relación con el no-show no es lineal.
    """
    transformers: list[tuple[str, Any, list[str]]] = [
        (
            "neighbourhood",
            Pipeline([("te", _neighbourhood_encoder()), ("sc", StandardScaler())]),
            ["neighbourhood"],
        ),
        ("age", _quantile_bins(), ["age"]),
        ("lead_time_days", _quantile_bins(), ["lead_time_days"]),
        (
            "appointment_weekday",
            OneHotEncoder(drop="first", sparse_output=False),
            ["appointment_weekday"],
        ),
        *[(name, StandardScaler(), [name]) for name in _SCALED],
        *[(name, "passthrough", [name]) for name in _PASSTHROUGH],
    ]
    pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])


def make_tree_preprocessor() -> ColumnTransformer:
    """LightGBM (comparación): barrio -> target encoding; el resto pasa sin cambios."""
    return ColumnTransformer(
        [("neighbourhood", _neighbourhood_encoder(), ["neighbourhood"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_lgbm(params: dict[str, float | int], n_estimators: int) -> LGBMClassifier:
    """LightGBM determinista con la semilla del proyecto (solo comparación)."""
    return LGBMClassifier(
        **params,  # type: ignore[arg-type]
        n_estimators=n_estimators,
        objective="binary",
        metric="average_precision",
        random_state=SEED,
        deterministic=True,
        force_row_wise=True,
        n_jobs=4,
        verbose=-1,
    )


class Calibrator(Protocol):
    """Mapa monótono de probabilidad cruda a probabilidad calibrada."""

    def predict(self, p: np.ndarray) -> np.ndarray: ...


@dataclass
class PlattCalibrator:
    """Regresión logística sobre el logit de la probabilidad cruda."""

    model: LogisticRegression

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, _EPS, 1 - _EPS)
        return np.log(p / (1 - p)).reshape(-1, 1)

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        return cls(LogisticRegression(C=1e6, max_iter=1000).fit(cls._logit(p), y))

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self._logit(p))[:, 1])


@dataclass
class IsotonicCalibrator:
    """Regresión isotónica (escalonada, no paramétrica)."""

    model: IsotonicRegression

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        return cls(iso.fit(p, y))

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(p))


def fit_calibrator(method: CalibrationMethod, p: np.ndarray, y: np.ndarray) -> Calibrator:
    """Ajusta el calibrador pedido."""
    if method == "isotonic":
        return IsotonicCalibrator.fit(p, y)
    return PlattCalibrator.fit(p, y)


@dataclass
class NoShowModel:
    """Artefacto final: regresión logística (entrenada en train) + calibrador (ajustado en val).

    ``background`` es la media de las columnas transformadas en train: la referencia de SHAP.
    """

    pipeline: Pipeline
    calibrator: Calibrator
    calibration_method: CalibrationMethod
    background: np.ndarray

    @classmethod
    def build(
        cls,
        pipeline: Pipeline,
        calibrator: Calibrator,
        method: CalibrationMethod,
        X_train: pd.DataFrame,
    ) -> NoShowModel:
        """Crea el artefacto calculando la referencia SHAP con los datos de train."""
        background = pipeline.named_steps["pre"].transform(X_train[list(FEATURE_COLUMNS)])
        return cls(pipeline, calibrator, method, np.asarray(background).mean(axis=0))

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidad del modelo sin calibrar."""
        return np.asarray(self.pipeline.predict_proba(X[list(FEATURE_COLUMNS)])[:, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidad calibrada de no-show."""
        return self.calibrator.predict(self.raw_proba(X))

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Valores SHAP exactos del modelo lineal, en log-odds sin calibrar.

        Para un modelo lineal, el SHAP de la columna transformada j es
        ``coef_j * (z_j - media_train_j)``. Las columnas que vienen de la misma feature original
        (por ejemplo, los 10 bins de edad) se suman. Cada fila más ``base_value`` da el log-odds.
        """
        pre = self.pipeline.named_steps["pre"]
        clf = self.pipeline.named_steps["clf"]
        Z = np.asarray(pre.transform(X[list(FEATURE_COLUMNS)]))
        coef = clf.coef_[0]
        phi = pd.DataFrame(
            (Z - self.background) * coef, columns=pre.get_feature_names_out(), index=X.index
        )
        source = [name.split("__", 1)[0] for name in phi.columns]
        per_feature = phi.T.groupby(source).sum().T
        out = per_feature.reindex(columns=list(FEATURE_COLUMNS), fill_value=0.0)
        out["base_value"] = float(clf.intercept_[0] + coef @ self.background)
        return out


def load_model(metadata_path: Path = PATHS.model_metadata) -> tuple[NoShowModel, dict[str, Any]]:
    """Carga el modelo principal indicado en ``metadata.json`` junto con su metadata."""
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = joblib.load(metadata_path.parent / metadata["model_file"])
    if not isinstance(model, NoShowModel):
        raise TypeError(f"{metadata['model_file']} no contiene un NoShowModel")
    return model, metadata
