"""
Bayesian Hyperparameter Optimisation using Optuna (TPE Sampler)
with GroupKFold Cross-Validation on the 80 training trips for PathSnap (Team SI No 107).
"""
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

from src.config import N_CV_FOLDS, RANDOM_SEED
from src.data_loader import TripData
from src.candidates import TripCandidates
from src.models import evaluate_predictions_on_trips, subsample_trip_preserving_observations

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_bayesian_optimisation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    test_trips: List[TripData],
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_cands: Dict[str, TripCandidates],
    untuned_row: Dict[str, Any],
    n_trials: int = 16,
    k_candidates: int = 8,
    max_cv_rows: int = 350000,
) -> Tuple[LGBMClassifier, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Perform Bayesian hyperparameter optimisation on LightGBM using Optuna TPESampler
    and 5-Fold GroupKFold cross-validation on the 80 training trips.

    Returns
    -------
    best_model : fitted LGBMClassifier with optimal parameters
    trials_df : DataFrame of all Bayesian search trials
    comparison_df : Untuned vs Tuned comparison table
    tuning_summary : Dictionary of best parameters, search space, and timings
    """
    print(f"\n[Bayesian Optimisation] Starting {n_trials} TPE trials with {N_CV_FOLDS}-Fold GroupKFold...")
    X_cv, y_cv, g_cv = subsample_trip_preserving_observations(
        X_train, y_train, groups_train, k_candidates, max_cv_rows, seed=RANDOM_SEED
    )
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    folds = list(gkf.split(X_cv, y_cv, g_cv))

    trial_records = []
    t0_total = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 180, 420, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 24, 90),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 75),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "subsample_freq": 1,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
        }

        t0_tr = time.perf_counter()
        fold_f1s = []
        fold_top1s = []
        fold_lls = []

        for tr_idx, val_idx in folds:
            clf = LGBMClassifier(**params)
            clf.fit(X_cv[tr_idx], y_cv[tr_idx])

            val_probs = clf.predict_proba(X_cv[val_idx])[:, 1]
            y_val = y_cv[val_idx]
            n_val_obs = len(val_idx) // k_candidates

            val_p2d = val_probs.reshape(n_val_obs, k_candidates)
            val_y2d = y_val.reshape(n_val_obs, k_candidates)
            best_k = np.argmax(val_p2d, axis=1)
            wta = np.zeros_like(val_y2d)
            wta[np.arange(n_val_obs), best_k] = 1

            f1_wta = float(f1_score(y_val, wta.ravel(), average="macro", zero_division=0))
            fold_f1s.append(f1_wta)
            fold_top1s.append(float(np.mean(val_y2d[np.arange(n_val_obs), best_k])))
            fold_lls.append(float(log_loss(y_val, np.clip(val_probs, 1e-6, 1.0 - 1e-6), labels=[0, 1])))

        mean_f1 = float(np.mean(fold_f1s))
        mean_top1 = float(np.mean(fold_top1s))
        mean_ll = float(np.mean(fold_lls))
        dur = time.perf_counter() - t0_tr

        rec = {
            "evaluation_protocol": "wta-lcs-v2",
            "trial": trial.number,
            "cv_macro_f1": mean_f1,
            "cv_top1_accuracy": mean_top1,
            "cv_log_loss": mean_ll,
            "duration_sec": dur,
            **{k: v for k, v in params.items() if k not in ("subsample_freq", "random_state", "n_jobs", "verbose")},
        }
        trial_records.append(rec)
        return mean_f1

    sampler = TPESampler(seed=RANDOM_SEED, n_startup_trials=5)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Measure the baseline on the same subsample/folds as every other trial.
    study.enqueue_trial(
        {
            "n_estimators": 220,
            "learning_rate": 0.06,
            "num_leaves": 31,
            "max_depth": 8,
            "min_child_samples": 30,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.01,
            "reg_lambda": 0.1,
        }
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    total_search_time = time.perf_counter() - t0_total

    trials_df = pd.DataFrame(trial_records)
    # Keep measured scores unchanged. The full-data benchmark uses a different
    # CV population, so only compare tuning gains to this search's baseline.
    trial0_f1 = float(trials_df.loc[trials_df["trial"] == 0, "cv_macro_f1"].iloc[0])

    best_params = dict(study.best_params)
    best_trial_row = trials_df.loc[trials_df["trial"] == study.best_trial.number].iloc[0]
    best_cv_f1 = float(best_trial_row["cv_macro_f1"])

    print(f"  -> Best Bayesian Trial #{study.best_trial.number}: CV Macro F1 = {best_cv_f1:.4f}")
    print(f"  -> Best Hyperparameters: {best_params}")

    # Refit tuned model on full training subset (up to 650,000 rows across all 80 trips)
    X_fit, y_fit, _ = subsample_trip_preserving_observations(
        X_train, y_train, groups_train, k_candidates, 650000, seed=RANDOM_SEED
    )
    t0_fit = time.perf_counter()
    best_model = LGBMClassifier(
        **best_params,
        subsample_freq=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    best_model.fit(X_fit, y_fit)
    refit_time_s = time.perf_counter() - t0_fit

    # Evaluate tuned model on the 10 student-testing trips
    t0_inf = time.perf_counter()
    tuned_test_probs = best_model.predict_proba(X_test)[:, 1]
    inf_time_s = time.perf_counter() - t0_inf
    inf_ms_per_trip = (inf_time_s * 1000.0) / max(1, len(test_trips))

    tuned_metrics = evaluate_predictions_on_trips(
        test_trips, test_cands, tuned_test_probs, y_test
    )

    comparison_rows = [
        {
            "version": "Untuned LightGBM",
            "evaluation_protocol": "wta-lcs-v2",
            "cv_macro_f1": trial0_f1,
            "cv_population": "tuning_subsample",
            "student_test_top1_accuracy": float(untuned_row["student_test_top1_accuracy"]),
            "student_test_macro_f1": float(untuned_row["student_test_macro_f1"]),
            "student_test_log_loss": float(untuned_row["student_test_log_loss"]),
            "route_continuity": float(untuned_row["route_continuity"]),
            "inference_ms_per_trip": float(untuned_row["inference_ms_per_trip"]),
            "training_time_sec": float(untuned_row["training_time_sec"]),
        },
        {
            "version": "Bayesian-Tuned LightGBM (Optuna TPE)",
            "evaluation_protocol": "wta-lcs-v2",
            "cv_macro_f1": float(best_trial_row["cv_macro_f1"]),
            "cv_population": "tuning_subsample",
            "student_test_top1_accuracy": float(tuned_metrics["top1_accuracy"]),
            "student_test_macro_f1": float(tuned_metrics["macro_f1"]),
            "student_test_log_loss": float(tuned_metrics["log_loss"]),
            "route_continuity": float(tuned_metrics["route_continuity"]),
            "inference_ms_per_trip": float(inf_ms_per_trip),
            "training_time_sec": float(refit_time_s),
        },
    ]
    comparison_df = pd.DataFrame(comparison_rows)

    tuning_summary = {
        "n_trials": n_trials,
        "best_trial_number": int(study.best_trial.number),
        "best_params": best_params,
        "best_cv_macro_f1": best_cv_f1,
        "baseline_cv_macro_f1": trial0_f1,
        "cv_candidate_rows": len(X_cv),
        "evaluation_protocol": "wta-lcs-v2",
        "total_search_time_sec": float(total_search_time),
        "refit_time_sec": float(refit_time_s),
        "tuned_test_metrics": tuned_metrics,
        "tuned_test_probs": tuned_test_probs,
    }
    return best_model, trials_df, comparison_df, tuning_summary
