"""Pipelines de modelos, calibradores y el artefacto final ``NoShowModel``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler, TargetEncoder

from noshow_guard.config import SEED
from noshow_guard.features import FEATURE_COLUMNS

CalibrationMethod = Literal["isotonic", "platt"]
_EPS = 1e-6


def _neighbourhood_encoder() -> TargetEncoder:
    """Target encoding suavizado; en ``fit_transform`` usa validación cruzada interna."""
    return TargetEncoder(target_type="binary", smooth="auto", cv=5, shuffle=True, random_state=SEED)


def make_tree_preprocessor() -> ColumnTransformer:
    """Barrio -> target encoding; el resto pasa sin cambios (los árboles no necesitan escalar)."""
    return ColumnTransformer(
        [("neighbourhood", _neighbourhood_encoder(), ["neighbourhood"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_lgbm(params: dict[str, float | int], n_estimators: int) -> LGBMClassifier:
    """LightGBM determinista con la semilla del proyecto."""
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


def make_logistic_pipeline() -> Pipeline:
    """Regresión logística: bins para relaciones no lineales (edad, antelación), escalado."""
    bins = KBinsDiscretizer(
        n_bins=10,
        encode="onehot-dense",
        strategy="quantile",
        quantile_method="averaged_inverted_cdf",
    )
    pre = ColumnTransformer(
        [
            (
                "neighbourhood",
                Pipeline([("te", _neighbourhood_encoder()), ("sc", StandardScaler())]),
                ["neighbourhood"],
            ),
            ("bins", bins, ["age", "lead_time_days"]),
            ("weekday", OneHotEncoder(drop="first"), ["appointment_weekday"]),
            (
                "scaled",
                StandardScaler(),
                ["handicap", "prev_appointments", "prev_no_shows", "prev_no_show_rate"],
            ),
        ],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])


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
    """Artefacto final: pipeline LightGBM (entrenado en train) + calibrador (ajustado en val)."""

    pipeline: Pipeline
    calibrator: Calibrator
    calibration_method: CalibrationMethod

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidad del modelo sin calibrar."""
        return np.asarray(self.pipeline.predict_proba(X[list(FEATURE_COLUMNS)])[:, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidad calibrada de no-show."""
        return self.calibrator.predict(self.raw_proba(X))

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Valores SHAP (TreeSHAP exacto de LightGBM) en log-odds del modelo sin calibrar.

        Columnas: una por feature original más ``base_value``. La fila suma el log-odds crudo.
        """
        pre = self.pipeline.named_steps["pre"]
        clf = self.pipeline.named_steps["clf"]
        Xt = pre.transform(X[list(FEATURE_COLUMNS)])
        values = clf.predict_proba(Xt, pred_contrib=True)
        frame = pd.DataFrame(values, columns=[*Xt.columns, "base_value"], index=X.index)
        return frame[[*FEATURE_COLUMNS, "base_value"]]
