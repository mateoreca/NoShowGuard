"""Entrenamiento y evaluación: baselines, regresión logística, LightGBM, calibración y umbrales.

Uso: ``python -m noshow_guard.train`` (o ``make train``). Lee los parquet de features y escribe:
- ``models/<fecha>_<hashdatos>.joblib`` (modelo principal) y ``models/metadata.json``
- ``reports/metrics.json`` y ``reports/figures/*.png``

Modelo principal: regresión logística calibrada. LightGBM se entrena como comparación
(ver ``MODEL_DECISION`` y docs/adr/0001-modelo-principal.md).

Reparto de datos: train ajusta los modelos; validación elige hiperparámetros de LightGBM (con
early stopping), ajusta el calibrador y elige los umbrales; test solo se usa para reportar.
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
from noshow_guard.config import (
    COSTS,
    PATHS,
    RAW_SHA256,
    SEARCH,
    SEED,
    CostConfig,
    Paths,
    SearchConfig,
)
from noshow_guard.evaluation import (
    ACTIONS,
    Thresholds,
    analytic_thresholds,
    assign_actions,
    binary_metrics,
    mean_cost,
    optimize_thresholds,
    risk_cutoffs,
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
PRINCIPAL_MODEL = "regresion_logistica_calibrada"

# Cortes de la comparación con origen móvil: (fin de train, fin de evaluación). Solo usan
# fechas de train + validación; test no participa.
ROLLING_CUTS: tuple[tuple[str, str], ...] = (
    ("2016-05-06", "2016-05-13"),
    ("2016-05-13", "2016-05-20"),
    ("2016-05-20", "2016-05-31"),
)

MODEL_DECISION: dict[str, Any] = {
    "principal": PRINCIPAL_MODEL,
    "comparacion": "lightgbm_calibrado",
    "decidido_despues_de_ver_test": True,
    "motivo": (
        "La regresión logística y LightGBM empatan en PR-AUC de validación (diferencia < 0,0001) "
        "y en los 3 cortes temporales con origen móvil (train + validación, sin test) no hay "
        "ganador consistente: diferencias de +0,007, -0,001 y 0,000 para LightGBM (cifras al "
        "momento de decidir, con conteos de historial lineales; ver ADR 0002). Ante un "
        "empate se prefiere el modelo más simple y explicable. La decisión se tomó después de "
        "ver test, donde LightGBM quedó por debajo; no se hizo otra selección de LightGBM "
        "contra test."
    ),
    "adr": "docs/adr/0001-modelo-principal.md",
}


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
) -> tuple[Pipeline, dict[str, float | int], pd.DataFrame]:
    """Búsqueda aleatoria; gana la mayor PR-AUC en validación.

    Devuelve el mejor pipeline, sus hiperparámetros y la tabla de intentos.
    """
    rng = np.random.default_rng(SEED)
    best: tuple[float, Pipeline, dict[str, float | int]] | None = None
    trials = []
    for i in range(search.n_iter):
        params = sample_params(rng)
        pipe, best_iter, val_ap = fit_lgbm(params, train, val, search)
        trials.append({"trial": i, **params, "best_iteration": best_iter, "val_pr_auc": val_ap})
        if best is None or val_ap > best[0]:
            best = (val_ap, pipe, params)
    assert best is not None
    return best[1], best[2], pd.DataFrame(trials).sort_values("val_pr_auc", ascending=False)


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


def calibrate(
    p_val_raw: np.ndarray, y_val: np.ndarray, folds: int
) -> tuple[CalibrationMethod, dict[str, float]]:
    """Elige isotónica o Platt por Brier fuera de fold dentro de validación."""
    cv = calibration_cv(p_val_raw, y_val, folds)
    method: CalibrationMethod = min(CALIBRATION_METHODS, key=lambda m: cv[m])
    return method, cv


# --- Comparación temporal ------------------------------------------------------


def rolling_origin_comparison(
    splits: dict[str, pd.DataFrame],
    lgbm_params: dict[str, float | int],
    search: SearchConfig,
    cuts: tuple[tuple[str, str], ...] = ROLLING_CUTS,
) -> list[dict[str, Any]]:
    """PR-AUC de regresión logística y LightGBM en cortes temporales sucesivos (sin test).

    LightGBM usa los mejores hiperparámetros de validación y hace early stopping en el
    propio bloque evaluado, lo que lo favorece levemente.
    """
    pool = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    day = pool["appointment_date"]
    rows = []
    for train_end, eval_end in cuts:
        fit_part = pool[day <= train_end]
        eval_part = pool[(day > train_end) & (day <= eval_end)]
        if fit_part.empty or eval_part.empty:
            continue
        fit_xy, eval_xy = xy(fit_part), xy(eval_part)
        lr = make_logistic_pipeline().fit(*fit_xy)
        lr_ap = average_precision_score(eval_xy[1], lr.predict_proba(eval_xy[0])[:, 1])
        _, _, lgbm_ap = fit_lgbm(lgbm_params, fit_xy, eval_xy, search)
        rows.append(
            {
                "train_hasta": train_end,
                "evalua_hasta": eval_end,
                "n_eval": len(eval_part),
                "regresion_logistica": float(lr_ap),
                "lightgbm": float(lgbm_ap),
            }
        )
    return rows


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


def _shap_display(model: NoShowModel, X: pd.DataFrame) -> pd.DataFrame:
    """Features numéricas para colorear el beeswarm (barrio -> su target encoding)."""
    te = model.pipeline.named_steps["pre"].named_transformers_["neighbourhood"].named_steps["te"]
    return X.assign(neighbourhood=te.transform(X[["neighbourhood"]]).ravel())


def fit_all(
    splits: dict[str, pd.DataFrame],
    costs: CostConfig = COSTS,
    search: SearchConfig = SEARCH,
) -> TrainResult:
    """Entrena baselines, el modelo principal y la comparación; elige umbrales; evalúa en test."""
    train, val, test = (xy(splits[n]) for n in SPLIT_NAMES)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = train, val, test

    # Baselines
    prior = float(y_tr.mean())
    k = rule_lead_threshold(X_tr["lead_time_days"].to_numpy(), y_tr)

    # Modelo principal: regresión logística calibrada en validación
    logistic = make_logistic_pipeline().fit(X_tr, y_tr)
    lr_val_raw = logistic.predict_proba(X_val)[:, 1]
    lr_method, lr_cal_cv = calibrate(lr_val_raw, y_val, search.calibration_folds)
    model = NoShowModel.build(
        logistic, fit_calibrator(lr_method, lr_val_raw, y_val), lr_method, X_tr
    )

    # Comparación: LightGBM (búsqueda en validación) calibrado con el mismo procedimiento
    lgbm, lgbm_params, trials = random_search(train, val, search)
    lgbm_val_raw = lgbm.predict_proba(X_val)[:, 1]
    lgbm_method, lgbm_cal_cv = calibrate(lgbm_val_raw, y_val, search.calibration_folds)
    lgbm_cal = fit_calibrator(lgbm_method, lgbm_val_raw, y_val)
    best = trials.iloc[0]

    # Umbrales por costo en validación, sobre la probabilidad calibrada del modelo principal
    p_val = model.predict_proba(X_val)
    thresholds, cost_table = optimize_thresholds(p_val, y_val, costs)

    # Evaluación en test (solo reporte)
    p_te_raw = model.raw_proba(X_te)
    p_te = model.predict_proba(X_te)
    lgbm_te_raw = lgbm.predict_proba(X_te)[:, 1]
    scores_te = {
        "clase_mayoritaria": np.full(len(y_te), prior),
        "regla_antelacion": X_te["lead_time_days"].to_numpy().astype(float),
        "regresion_logistica_sin_calibrar": p_te_raw,
        PRINCIPAL_MODEL: p_te,
        "lightgbm_sin_calibrar": lgbm_te_raw,
        "lightgbm_calibrado": lgbm_cal.predict(lgbm_te_raw),
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
    shap_rows = splits["test"].iloc[shap_idx]
    contrib = model.contributions(shap_rows)

    metrics: dict[str, Any] = {
        "principal_model": PRINCIPAL_MODEL,
        "model_decision": MODEL_DECISION,
        "rows": {n: len(splits[n]) for n in SPLIT_NAMES},
        "test": model_metrics,
        "val_pr_auc": {
            "regresion_logistica_sin_calibrar": float(average_precision_score(y_val, lr_val_raw)),
            "lightgbm_sin_calibrar": float(best["val_pr_auc"]),
        },
        "rolling_origin_pr_auc": rolling_origin_comparison(splits, lgbm_params, search),
        "rule_lead_k": k,
        "lgbm_search": {
            "n_trials": len(trials),
            "best_params": lgbm_params,
            "best_iteration": int(best["best_iteration"]),
        },
        "calibration": {
            "method": lr_method,
            "cv_brier_val": lr_cal_cv,
            "lightgbm": {"method": lgbm_method, "cv_brier_val": lgbm_cal_cv},
        },
        "costs": asdict(costs),
        "thresholds": thresholds.to_dict(),
        "analytic_thresholds": analytic_thresholds(costs).to_dict(),
        "policy_cost_per_appointment": {
            "val": _policy_costs(p_val, y_val, thresholds, costs),
            "test": _policy_costs(p_te, y_te, thresholds, costs),
        },
        "risk_cutoffs": risk_cutoffs(p_val),
        "history_counts_transform": "log1p (ADR 0002)",
        "action_counts_test": {
            action: int(np.sum(assign_actions(p_te, thresholds) == i))
            for i, action in enumerate(ACTIONS)
        },
        "max_probability": {"val": float(p_val.max()), "test": float(p_te.max())},
        "decision_metrics_test": {
            "any_reminder": binary_metrics(y_te, p_te >= thresholds.standard),
            "reinforced": binary_metrics(y_te, p_te >= thresholds.reinforced),
        },
        "segments_test": segments,
        "shap": {
            "method": "lineal exacto, agregado por feature original (log-odds sin calibrar)",
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
        "calibration_test": {
            "reg. logística sin calibrar": p_te_raw,
            f"reg. logística calibrada ({lr_method})": p_te,
        },
        "shap_contrib": contrib,
        "shap_features": _shap_display(model, shap_rows[list(FEATURE_COLUMNS)]),
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


def build_metadata(
    metrics: dict[str, Any], version_name: str, dhash: str, created_at: str
) -> dict[str, Any]:
    """Metadata que acompaña al modelo: la leen el simulador y las acciones."""
    return {
        "model_version": version_name,
        "model_file": f"{version_name}.joblib",
        "created_at": created_at,
        "data_hash": dhash,
        "raw_sha256": RAW_SHA256,
        "seed": SEED,
        "features": list(FEATURE_COLUMNS),
        "principal_model": PRINCIPAL_MODEL,
        "model_decision": MODEL_DECISION,
        "calibration_method": metrics["calibration"]["method"],
        "thresholds": metrics["thresholds"],
        "analytic_thresholds": metrics["analytic_thresholds"],
        "actions": list(ACTIONS),
        "costs": metrics["costs"],
        "costs_are_assumptions": True,
        "risk_cutoffs": metrics["risk_cutoffs"],
        "train_ranges": metrics["train_ranges"],
        "test_metrics": {k: round(v, 4) for k, v in metrics["test"][PRINCIPAL_MODEL].items()},
        "libraries": {
            lib: version(lib) for lib in ("scikit-learn", "lightgbm", "pandas", "numpy", "shap")
        },
    }


def save(result: TrainResult, splits: dict[str, pd.DataFrame], paths: Paths = PATHS) -> str:
    """Guarda modelo, metadata, métricas y figuras. Devuelve la versión del modelo."""
    dhash = data_hash(splits)
    created = datetime.now(UTC)
    version_name = f"{created:%Y%m%d}_{dhash[:8]}"
    paths.models.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, paths.models / f"{version_name}.joblib")

    m = result.metrics
    metadata = build_metadata(m, version_name, dhash, created.isoformat(timespec="seconds"))
    paths.model_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    paths.metrics.write_text(
        json.dumps(_json_safe({"model_version": version_name, **m}), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    result.trials.to_csv(paths.reports / "search_trials.csv", index=False)

    fig = paths.figures
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
    print("Umbrales:", result.thresholds, "| calibración:", result.metrics["calibration"]["method"])
    print("Origen móvil:", json.dumps(result.metrics["rolling_origin_pr_auc"], indent=1))
    print(f"Modelo guardado: models/{name}.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
