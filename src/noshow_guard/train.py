"""Entrenamiento y evaluación: baselines, regresión logística, LightGBM, calibración y umbrales.

Uso: ``python -m noshow_guard.train`` (o ``make train``). Lee los parquet de features y escribe:
- ``models/<fecha>_<hashdatos>.joblib`` y ``models/metadata.json``
- ``reports/metrics.json`` y ``reports/figures/*.png``

Reparto de datos: train ajusta los modelos; validación elige hiperparámetros (con early
stopping), ajusta el calibrador y elige los umbrales; test solo se usa para reportar.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from noshow_guard import plots
from noshow_guard.config import COSTS, PATHS, RAW_SHA256, SEARCH, SEED, CostConfig, SearchConfig
from noshow_guard.evaluation import (
    Thresholds,
    analytic_thresholds,
    assign_actions,
    binary_metrics,
    mean_cost,
    optimize_thresholds,
    score_metrics,
    segment_table,
)
from noshow_guard.explain import global_importance, top_factors
from noshow_guard.features import FEATURE_COLUMNS, SPLIT_NAMES
from noshow_guard.model import (
    CalibrationMethod,
    NoShowModel,
    fit_calibrator,
    make_lgbm,
    make_logistic_pipeline,
    make_tree_preprocessor,
)

PARAM_SPACE: dict[str, list[float | int]] = {
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "num_leaves": [7, 15, 31, 63],
    "min_child_samples": [20, 50, 100, 200, 400],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0, 20.0],
}
CALIBRATION_METHODS: tuple[CalibrationMethod, ...] = ("isotonic", "platt")
AGE_BINS = ([-1, 17, 35, 55, 75, 200], ["0-17", "18-35", "36-55", "56-75", "76+"])
LEAD_BINS = ([0, 2, 7, 14, 30, 10_000], ["1-2", "3-7", "8-14", "15-30", "31+"])
SHAP_SAMPLE = 3000


@dataclass
class TrainResult:
    """Todo lo que produce un entrenamiento, antes de guardarlo en disco."""

    model: NoShowModel
    thresholds: Thresholds
    metrics: dict[str, Any]
    trials: pd.DataFrame
    cost_table: pd.DataFrame
    curves: dict[str, Any] = field(default_factory=dict)


# --- Datos ---------------------------------------------------------------------


def load_splits() -> dict[str, pd.DataFrame]:
    """Lee los parquet de features generados por ``make features``."""
    return {name: pd.read_parquet(PATHS.features_dir / f"{name}.parquet") for name in SPLIT_NAMES}


def xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Features en el orden oficial y objetivo."""
    return frame[list(FEATURE_COLUMNS)], frame["no_show"].to_numpy()


def data_hash(splits: dict[str, pd.DataFrame]) -> str:
    """Hash estable del contenido de los tres splits (features + objetivo)."""
    digest = hashlib.sha256()
    for name in SPLIT_NAMES:
        cols = ["appointment_id", "no_show", *FEATURE_COLUMNS]
        hashes = pd.util.hash_pandas_object(splits[name][cols], index=False)
        digest.update(hashes.to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()


# --- Baselines -----------------------------------------------------------------


def rule_lead_threshold(lead: np.ndarray, y: np.ndarray, max_days: int = 60) -> int:
    """Regla "antelación alta": k (1..max_days) que maximiza F1 en train para ``lead >= k``."""

    def f1(k: int) -> float:
        m = binary_metrics(y, lead >= k)
        p, r = m["precision"], m["recall"]
        return 0.0 if not p or not r or np.isnan(p) else 2 * p * r / (p + r)

    return max(range(1, max_days + 1), key=f1)


# --- LightGBM ------------------------------------------------------------------


def sample_params(rng: np.random.Generator) -> dict[str, float | int]:
    """Una configuración aleatoria del espacio de búsqueda."""
    params = {name: rng.choice(np.array(values)).item() for name, values in PARAM_SPACE.items()}
    return {**params, "subsample_freq": 1}


def fit_lgbm(
    params: dict[str, float | int],
    train: tuple[pd.DataFrame, np.ndarray],
    val: tuple[pd.DataFrame, np.ndarray],
    search: SearchConfig,
) -> tuple[Pipeline, int, float]:
    """Entrena en train con early stopping en val.

    Devuelve el pipeline, la mejor iteración y la PR-AUC en validación.
    """
    pre = make_tree_preprocessor()
    X_train = pre.fit_transform(train[0], train[1])
    X_val = pre.transform(val[0])
    clf = make_lgbm(params, search.max_estimators)
    clf.fit(
        X_train,
        train[1],
        eval_set=[(X_val, val[1])],
        callbacks=[lgb.early_stopping(search.early_stopping_rounds, verbose=False)],
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    val_ap = float(average_precision_score(val[1], pipe.predict_proba(val[0])[:, 1]))
    return pipe, int(clf.best_iteration_), val_ap


def random_search(
    train: tuple[pd.DataFrame, np.ndarray],
    val: tuple[pd.DataFrame, np.ndarray],
    search: SearchConfig,
) -> tuple[Pipeline, pd.DataFrame]:
    """Búsqueda aleatoria; gana la mayor PR-AUC en validación."""
    rng = np.random.default_rng(SEED)
    best: tuple[float, Pipeline] | None = None
    trials = []
    for i in range(search.n_iter):
        params = sample_params(rng)
        pipe, best_iter, val_ap = fit_lgbm(params, train, val, search)
        trials.append({"trial": i, **params, "best_iteration": best_iter, "val_pr_auc": val_ap})
        if best is None or val_ap > best[0]:
            best = (val_ap, pipe)
    assert best is not None
    return best[1], pd.DataFrame(trials).sort_values("val_pr_auc", ascending=False)


# --- Calibración ---------------------------------------------------------------


def calibration_cv(p_val: np.ndarray, y_val: np.ndarray, folds: int) -> dict[str, float]:
    """Brier fuera de fold dentro de validación para cada método (y sin calibrar)."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    scores: dict[str, list[float]] = {"raw": [], **{m: [] for m in CALIBRATION_METHODS}}
    for fit_idx, eval_idx in skf.split(p_val, y_val):
        scores["raw"].append(brier_score_loss(y_val[eval_idx], p_val[eval_idx]))
        for method in CALIBRATION_METHODS:
            cal = fit_calibrator(method, p_val[fit_idx], y_val[fit_idx])
            scores[method].append(brier_score_loss(y_val[eval_idx], cal.predict(p_val[eval_idx])))
    return {name: float(np.mean(values)) for name, values in scores.items()}


# --- Orquestación --------------------------------------------------------------


def _segments(frame: pd.DataFrame, train_patients: set[str]) -> dict[str, pd.Series]:
    """Grupos para análisis de errores y equidad."""
    return {
        "historial": frame["has_history"].map({0: "sin historial", 1: "con historial"}),
        "paciente_en_train": frame["patient_id"]
        .isin(train_patients)
        .map({False: "no", True: "si"}),
        "edad": pd.cut(frame["age"], bins=AGE_BINS[0], labels=AGE_BINS[1]),
        "antelacion": pd.cut(frame["lead_time_days"], bins=LEAD_BINS[0], labels=LEAD_BINS[1]),
        "genero": frame["is_male"].map({0: "F", 1: "M"}),
        "beca": frame["scholarship"].map({0: "no", 1: "si"}),
    }


def _policy_costs(
    p: np.ndarray, y: np.ndarray, th: Thresholds, costs: CostConfig
) -> dict[str, float]:
    """Costo medio por cita: política del modelo vs políticas triviales."""
    return {
        "politica_modelo": mean_cost(p, y, th, costs),
        "no_hacer_nada": mean_cost(p, y, Thresholds(2.0, 2.0), costs),
        "estandar_a_todos": mean_cost(p, y, Thresholds(0.0, 2.0), costs),
        "reforzado_a_todos": mean_cost(p, y, Thresholds(0.0, 0.0), costs),
    }


def _local_examples(model: NoShowModel, frame: pd.DataFrame, p: np.ndarray) -> list[dict[str, Any]]:
    """Tres casos de test (probabilidad más alta, mediana y más baja) con sus factores SHAP."""
    order = np.argsort(p, kind="stable")
    picks = {"alta": order[-1], "mediana": order[len(order) // 2], "baja": order[0]}
    rows = frame.iloc[list(picks.values())]
    contrib = model.contributions(rows)
    return [
        {
            "caso": label,
            "probability_no_show": round(float(p[idx]), 4),
            "no_show_real": int(frame.iloc[idx]["no_show"]),
            "top_factors": top_factors(contrib.iloc[i], rows.iloc[i]),
        }
        for i, (label, idx) in enumerate(picks.items())
    ]


def fit_all(
    splits: dict[str, pd.DataFrame],
    costs: CostConfig = COSTS,
    search: SearchConfig = SEARCH,
) -> TrainResult:
    """Entrena todos los modelos, calibra, elige umbrales y evalúa en test."""
    train, val, test = (xy(splits[n]) for n in SPLIT_NAMES)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = train, val, test

    # Baselines
    prior = float(y_tr.mean())
    k = rule_lead_threshold(X_tr["lead_time_days"].to_numpy(), y_tr)
    logistic = make_logistic_pipeline().fit(X_tr, y_tr)

    # LightGBM + calibración
    pipe, trials = random_search(train, val, search)
    p_val_raw = pipe.predict_proba(X_val)[:, 1]
    cal_cv = calibration_cv(p_val_raw, y_val, search.calibration_folds)
    method: CalibrationMethod = min(CALIBRATION_METHODS, key=lambda m: cal_cv[m])
    model = NoShowModel(pipe, fit_calibrator(method, p_val_raw, y_val), method)

    # Umbrales por costo en validación
    p_val = model.predict_proba(X_val)
    thresholds, cost_table = optimize_thresholds(p_val, y_val, costs)

    # Evaluación en test
    p_te_raw = model.raw_proba(X_te)
    p_te = model.predict_proba(X_te)
    scores_te = {
        "clase_mayoritaria": np.full(len(y_te), prior),
        "regla_antelacion": X_te["lead_time_days"].to_numpy().astype(float),
        "regresion_logistica": logistic.predict_proba(X_te)[:, 1],
        "lightgbm_sin_calibrar": p_te_raw,
        "lightgbm_calibrado": p_te,
    }
    model_metrics = {name: score_metrics(y_te, s) for name, s in scores_te.items()}
    model_metrics["clase_mayoritaria"].update(binary_metrics(y_te, np.zeros(len(y_te))))
    rule_flags = (X_te["lead_time_days"] >= k).to_numpy()
    # La regla da un puntaje en días, no una probabilidad: score_metrics omite Brier y log-loss.
    model_metrics["regla_antelacion"].update(binary_metrics(y_te, rule_flags))

    train_patients = set(splits["train"]["patient_id"])
    segments = {
        name: segment_table(y_te, p_te, groups, thresholds.standard).to_dict("records")
        for name, groups in _segments(splits["test"], train_patients).items()
    }

    rng = np.random.default_rng(SEED)
    shap_idx = rng.choice(len(X_te), size=min(SHAP_SAMPLE, len(X_te)), replace=False)
    contrib = model.contributions(splits["test"].iloc[shap_idx])

    best = trials.iloc[0]
    metrics: dict[str, Any] = {
        "rows": {n: len(splits[n]) for n in SPLIT_NAMES},
        "test": model_metrics,
        "val_pr_auc": {
            "regresion_logistica": float(
                average_precision_score(y_val, logistic.predict_proba(X_val)[:, 1])
            ),
            "lightgbm_sin_calibrar": float(best["val_pr_auc"]),
        },
        "rule_lead_k": k,
        "search": {
            "n_trials": len(trials),
            "best_params": {p: best[p].item() for p in [*PARAM_SPACE, "best_iteration"]},
        },
        "calibration": {"method": method, "cv_brier_val": cal_cv},
        "costs": asdict(costs),
        "thresholds": thresholds.to_dict(),
        "analytic_thresholds": analytic_thresholds(costs).to_dict(),
        "policy_cost_per_appointment": {
            "val": _policy_costs(p_val, y_val, thresholds, costs),
            "test": _policy_costs(p_te, y_te, thresholds, costs),
        },
        "action_share_test": {
            str(a): float(np.mean(assign_actions(p_te, thresholds) == a)) for a in (0, 1, 2)
        },
        "decision_metrics_test": {
            "any_reminder": binary_metrics(y_te, p_te >= thresholds.standard),
            "reinforced": binary_metrics(y_te, p_te >= thresholds.reinforced),
        },
        "segments_test": segments,
        "shap": {
            "sample_size": len(shap_idx),
            "global_mean_abs": global_importance(contrib).round(4).to_dict(),
            "local_examples": _local_examples(model, splits["test"], p_te),
        },
        "train_ranges": {
            "lead_time_days_max": int(X_tr["lead_time_days"].max()),
            "age_min": int(X_tr["age"].min()),
            "age_max": int(X_tr["age"].max()),
        },
    }
    curves = {
        "y_test": y_te,
        "scores_test": {k_: v for k_, v in scores_te.items() if k_ != "clase_mayoritaria"},
        "calibration_test": {"sin calibrar": p_te_raw, f"calibrado ({method})": p_te},
        "shap_contrib": contrib,
        "shap_features": model.pipeline.named_steps["pre"].transform(
            splits["test"].iloc[shap_idx][list(FEATURE_COLUMNS)]
        )[list(FEATURE_COLUMNS)],
    }
    return TrainResult(model, thresholds, metrics, trials, cost_table, curves)


def _json_safe(obj: Any) -> Any:
    """Tipos numpy a nativos y NaN a ``null`` (NaN no es JSON válido)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save(result: TrainResult, splits: dict[str, pd.DataFrame]) -> str:
    """Guarda modelo, metadata, métricas y figuras. Devuelve la versión del modelo."""
    dhash = data_hash(splits)
    created = datetime.now(UTC)
    version_name = f"{created:%Y%m%d}_{dhash[:8]}"
    PATHS.models.mkdir(parents=True, exist_ok=True)
    PATHS.reports.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, PATHS.models / f"{version_name}.joblib")

    m = result.metrics
    metadata = {
        "model_version": version_name,
        "model_file": f"{version_name}.joblib",
        "created_at": created.isoformat(timespec="seconds"),
        "data_hash": dhash,
        "raw_sha256": RAW_SHA256,
        "seed": SEED,
        "features": list(FEATURE_COLUMNS),
        "calibration_method": m["calibration"]["method"],
        "thresholds": m["thresholds"],
        "costs": m["costs"],
        "best_params": m["search"]["best_params"],
        "train_ranges": m["train_ranges"],
        "test_metrics": {k: round(v, 4) for k, v in m["test"]["lightgbm_calibrado"].items()},
        "libraries": {
            lib: version(lib) for lib in ("scikit-learn", "lightgbm", "pandas", "numpy", "shap")
        },
    }
    PATHS.model_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    PATHS.metrics.write_text(
        json.dumps(_json_safe({"model_version": version_name, **m}), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    result.trials.to_csv(PATHS.reports / "search_trials.csv", index=False)

    fig = PATHS.figures
    c = result.curves
    plots.pr_curves(c["y_test"], c["scores_test"], fig / "pr_curves_test.png")
    plots.calibration_plot(c["y_test"], c["calibration_test"], fig / "calibration_test.png")
    val_costs = m["policy_cost_per_appointment"]["val"]
    plots.cost_curve(
        result.cost_table,
        result.thresholds,
        {
            "no hacer nada": val_costs["no_hacer_nada"],
            "estándar a todos": val_costs["estandar_a_todos"],
            "reforzado a todos": val_costs["reforzado_a_todos"],
        },
        fig / "cost_vs_threshold_val.png",
    )
    plots.shap_summary(c["shap_contrib"], c["shap_features"], fig / "shap")
    return version_name


def main() -> int:
    splits = load_splits()
    result = fit_all(splits)
    name = save(result, splits)
    summary = {k: {m: round(v, 4) for m, v in d.items()} for k, d in result.metrics["test"].items()}
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))
    print("Umbrales:", result.thresholds, "| calibración:", result.metrics["calibration"])
    print(f"Modelo guardado: models/{name}.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
