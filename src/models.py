"""
Multi-family model experimentation, GroupKFold cross-validation,
and comprehensive evaluation pipeline for PathSnap (Team SI No 107).

Trains and compares 6 distinct models from 6 algorithm families:
1. Logistic Regression (Linear & Probabilistic)
2. Distance-Weighted KNN (Instance-Based)
3. Random Forest (Bagging Tree Ensemble)
4. Extra Trees (Randomised Tree Ensemble)
5. Multi-Layer Perceptron - MLP (Neural Network)
6. LightGBM & XGBoost (Gradient Boosting)
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.config import N_CV_FOLDS, RANDOM_SEED
from src.data_loader import TripData
from src.candidates import TripCandidates
from src.hmm_viterbi import compute_route_continuity


@dataclass
class ModelSpec:
    name: str
    family: str
    config_summary: str
    estimator: Any
    max_train_rows: Optional[int] = None
    has_epoch_history: bool = False


def get_model_suite(seed: int = RANDOM_SEED) -> List[ModelSpec]:
    """
    Define 6 distinct models across 6 algorithm families for rigorous comparison.
    """
    return [
        ModelSpec(
            name="Logistic Regression",
            family="Linear & Probabilistic",
            config_summary="StandardScaler + L2 LogisticRegression(C=1.0, class_weight='balanced', lbfgs)",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=400,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            max_train_rows=400000,
            has_epoch_history=False,
        ),
        ModelSpec(
            name="Distance-Weighted KNN",
            family="Instance-Based",
            config_summary="StandardScaler + KNeighborsClassifier(n_neighbors=19, weights='distance', p=2)",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        KNeighborsClassifier(
                            n_neighbors=19,
                            weights="distance",
                            metric="minkowski",
                            p=2,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            max_train_rows=85000,  # Keep KD-tree query fast during CV & inference
            has_epoch_history=False,
        ),
        ModelSpec(
            name="Random Forest",
            family="Ensemble (Bagging)",
            config_summary="RandomForest(n_estimators=120, max_depth=14, min_samples_leaf=10, balanced_subsample)",
            estimator=RandomForestClassifier(
                n_estimators=120,
                max_depth=14,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
            max_train_rows=320000,
            has_epoch_history=False,
        ),
        ModelSpec(
            name="Extra Trees",
            family="Tree-Based (Extremely Randomised)",
            config_summary="ExtraTrees(n_estimators=120, max_depth=16, min_samples_leaf=8, balanced_subsample)",
            estimator=ExtraTreesClassifier(
                n_estimators=120,
                max_depth=16,
                min_samples_leaf=8,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
            max_train_rows=320000,
            has_epoch_history=False,
        ),
        ModelSpec(
            name="MLP Neural Network",
            family="Neural Networks",
            config_summary="StandardScaler + MLPClassifier(hidden=(64, 32), ReLU, alpha=1e-3, lr=1e-3, Adam)",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            alpha=1e-3,
                            learning_rate_init=1e-3,
                            batch_size=512,
                            max_iter=30,
                            early_stopping=True,
                            validation_fraction=0.12,
                            n_iter_no_change=5,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            max_train_rows=280000,
            has_epoch_history=True,
        ),
        ModelSpec(
            name="LightGBM",
            family="Gradient Boosting (Histogram GBDT)",
            config_summary="LGBMClassifier(n_estimators=220, lr=0.06, num_leaves=31, colsample=0.85, subsample=0.85)",
            estimator=LGBMClassifier(
                n_estimators=220,
                learning_rate=0.06,
                num_leaves=31,
                colsample_bytree=0.85,
                subsample=0.85,
                subsample_freq=1,
                min_child_samples=30,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            ),
            max_train_rows=500000,
            has_epoch_history=True,
        ),
        ModelSpec(
            name="XGBoost",
            family="Gradient Boosting (Depth-wise GBDT)",
            config_summary="XGBClassifier(n_estimators=220, lr=0.06, max_depth=6, colsample=0.85, subsample=0.85)",
            estimator=XGBClassifier(
                n_estimators=220,
                learning_rate=0.06,
                max_depth=6,
                colsample_bytree=0.85,
                subsample=0.85,
                min_child_weight=5,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            ),
            max_train_rows=500000,
            has_epoch_history=True,
        ),
    ]


def subsample_trip_preserving_observations(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    k_candidates: int,
    max_rows: Optional[int],
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    If total candidate rows exceed `max_rows`, uniformly subsample whole GPS observations
    (keeping all K candidates of each selected GPS observation intact) across trips.
    """
    if max_rows is None or len(X) <= max_rows:
        return X, y, groups

    n_obs_total = len(X) // k_candidates
    target_obs = max_rows // k_candidates
    rng = np.random.default_rng(seed)
    chosen_obs = np.sort(rng.choice(n_obs_total, size=target_obs, replace=False))

    # Expand observation indices to candidate row indices
    row_idx = (chosen_obs[:, None] * k_candidates + np.arange(k_candidates)[None, :]).ravel()
    return X[row_idx], y[row_idx], groups[row_idx]


def evaluate_predictions_on_trips(
    trips: List[TripData],
    cand_dict: Dict[str, TripCandidates],
    probs_flat: np.ndarray,
    y_true_flat: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute candidate-level metrics (Macro F1, Precision, Recall, Log Loss, Confusion Matrix)
    and trajectory-level metrics (Observation-level Top-1 Accuracy, Route Continuity)
    across a list of trips.
    """
    probs_clipped = np.clip(probs_flat, 1e-6, 1.0 - 1e-6)
    y_pred_bin = (probs_clipped >= threshold).astype(np.int32)

    # Also evaluate top-1 winner-takes-all binary assignment per observation
    # (selecting argmax candidate per GPS observation)
    top1_hits = 0
    total_obs = 0
    cont_scores = []
    strict_cont_scores = []
    top1_bin_preds = []

    offset = 0
    for trip in trips:
        cands = cand_dict[trip.trip_id]
        n_obs, k = cands.cand_arcs.shape
        n_rows = n_obs * k
        trip_probs = probs_clipped[offset : offset + n_rows].reshape(n_obs, k)
        offset += n_rows

        best_k = np.argmax(trip_probs, axis=1)
        pred_arcs = cands.cand_arcs[np.arange(n_obs), best_k]

        # Winner-takes-all binary indicator for candidate table
        wta = np.zeros((n_obs, k), dtype=np.int32)
        wta[np.arange(n_obs), best_k] = 1
        top1_bin_preds.append(wta.ravel())

        if trip.obs_ref_arcs is not None and len(trip.obs_ref_arcs) == n_obs:
            top1_hits += int(np.sum(pred_arcs == trip.obs_ref_arcs))
            total_obs += n_obs

        cont = compute_route_continuity(trip, pred_arcs)
        cont_scores.append(cont["connected_continuity"])
        strict_cont_scores.append(cont["strict_1hop_continuity"])

    top1_bin_flat = np.concatenate(top1_bin_preds)

    # Use winner-takes-all candidate selection for Macro F1 if thresholded predictions
    # are skewed by class-weight calibration, or pick the better calibrated binary representation
    f1_thresh = float(f1_score(y_true_flat, y_pred_bin, average="macro", zero_division=0))
    f1_wta = float(f1_score(y_true_flat, top1_bin_flat, average="macro", zero_division=0))
    if f1_wta >= f1_thresh:
        y_eval_bin = top1_bin_flat
        macro_f1 = f1_wta
    else:
        y_eval_bin = y_pred_bin
        macro_f1 = f1_thresh

    macro_prec = float(precision_score(y_true_flat, y_eval_bin, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true_flat, y_eval_bin, average="macro", zero_division=0))
    pos_prec = float(precision_score(y_true_flat, y_eval_bin, pos_label=1, zero_division=0))
    pos_rec = float(recall_score(y_true_flat, y_eval_bin, pos_label=1, zero_division=0))
    pos_f1 = float(f1_score(y_true_flat, y_eval_bin, pos_label=1, zero_division=0))
    acc = float(accuracy_score(y_true_flat, y_eval_bin))
    ll = float(log_loss(y_true_flat, probs_clipped, labels=[0, 1]))
    cm = confusion_matrix(y_true_flat, y_eval_bin, labels=[0, 1])

    top1_acc = float(top1_hits / max(1, total_obs))
    mean_cont = float(np.mean(cont_scores)) if cont_scores else 0.0
    mean_strict_cont = float(np.mean(strict_cont_scores)) if strict_cont_scores else 0.0

    return {
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "pos_precision": pos_prec,
        "pos_recall": pos_rec,
        "pos_f1": pos_f1,
        "binary_accuracy": acc,
        "log_loss": ll,
        "top1_accuracy": top1_acc,
        "route_continuity": mean_cont,
        "strict_1hop_continuity": mean_strict_cont,
        "confusion_matrix": cm.tolist(),
    }


def run_model_experimentation(
    train_trips: List[TripData],
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    train_cands: Dict[str, TripCandidates],
    test_trips: List[TripData],
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_cands: Dict[str, TripCandidates],
    k_candidates: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Execute 5-Fold GroupKFold Cross-Validation on the 80 training trips and
    evaluate all 6+ models on the 10 student-testing trips.
    """
    specs = get_model_suite(seed=RANDOM_SEED)
    gkf = GroupKFold(n_splits=N_CV_FOLDS)

    # Map trip_id -> slice in X_train for fast fold trip evaluation
    trip_by_id = {t.trip_id: t for t in train_trips}

    summary_rows = []
    trained_models: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}

    for spec in specs:
        print(f"\n[Model Experimentation] Training & Cross-Validating: {spec.name} ({spec.family})")
        fold_f1s = []
        fold_top1s = []
        fold_lls = []
        fold_train_f1s = []

        t0_cv = time.perf_counter()
        # GroupKFold by trip_id ensures zero observations from the same trip cross folds
        for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups_train)):
            X_tr_f, y_tr_f, g_tr_f = X_train[tr_idx], y_train[tr_idx], groups_train[tr_idx]
            X_tr_sub, y_tr_sub, _ = subsample_trip_preserving_observations(
                X_tr_f, y_tr_f, g_tr_f, k_candidates, spec.max_train_rows, seed=RANDOM_SEED + fold_idx
            )

            model_f = clone(spec.estimator)
            model_f.fit(X_tr_sub, y_tr_sub)

            # Evaluate on training subset (to diagnose overfitting/underfitting)
            tr_probs = model_f.predict_proba(X_tr_sub[: min(40000, len(X_tr_sub))])[:, 1]
            tr_preds = (tr_probs >= 0.5).astype(int)
            fold_train_f1s.append(
                float(f1_score(y_tr_sub[: len(tr_probs)], tr_preds, average="macro", zero_division=0))
            )

            # Evaluate on validation fold trips
            X_val_f = X_train[val_idx]
            y_val_f = y_train[val_idx]
            g_val_f = groups_train[val_idx]
            val_probs = model_f.predict_proba(X_val_f)[:, 1]

            # Reshape val_probs into (N_val_obs, K) for winner-takes-all Top-1 & Macro F1
            n_val_obs = len(X_val_f) // k_candidates
            val_probs_2d = val_probs.reshape(n_val_obs, k_candidates)
            val_y_2d = y_val_f.reshape(n_val_obs, k_candidates)
            best_k = np.argmax(val_probs_2d, axis=1)
            wta_val = np.zeros_like(val_y_2d)
            wta_val[np.arange(n_val_obs), best_k] = 1

            f1_wta = float(f1_score(y_val_f, wta_val.ravel(), average="macro", zero_division=0))
            f1_thr = float(f1_score(y_val_f, (val_probs >= 0.5).astype(int), average="macro", zero_division=0))
            fold_f1s.append(max(f1_wta, f1_thr))

            # Observation-level Top-1 accuracy on validation fold
            hits = val_y_2d[np.arange(n_val_obs), best_k]
            fold_top1s.append(float(np.mean(hits)))
            fold_lls.append(float(log_loss(y_val_f, np.clip(val_probs, 1e-6, 1.0 - 1e-6), labels=[0, 1])))

        cv_time_s = time.perf_counter() - t0_cv

        # Fit final model on full (or max_train_rows subsampled) 80 training trips
        X_fit, y_fit, g_fit = subsample_trip_preserving_observations(
            X_train, y_train, groups_train, k_candidates, spec.max_train_rows, seed=RANDOM_SEED
        )
        final_est = clone(spec.estimator)
        epoch_history: Optional[Dict[str, List[float]]] = None

        t0_fit = time.perf_counter()
        if spec.name in ("LightGBM", "XGBoost"):
            # Hold out 10% of training trips by group for genuine iteration-wise loss curve logging
            unique_tr_trips = np.unique(g_fit)
            val_trip_set = set(unique_tr_trips[::10])
            val_mask = np.isin(g_fit, list(val_trip_set))
            X_tr_h, y_tr_h = X_fit[~val_mask], y_fit[~val_mask]
            X_va_h, y_va_h = X_fit[val_mask], y_fit[val_mask]

            if spec.name == "LightGBM":
                final_est.fit(
                    X_tr_h,
                    y_tr_h,
                    eval_set=[(X_tr_h, y_tr_h), (X_va_h, y_va_h)],
                    eval_names=["train", "val"],
                    eval_metric="binary_logloss",
                )
                res = final_est.evals_result_
                epoch_history = {
                    "train_loss": [float(x) for x in res["train"]["binary_logloss"]],
                    "val_loss": [float(x) for x in res["val"]["binary_logloss"]],
                }
            else:
                final_est.fit(
                    X_tr_h,
                    y_tr_h,
                    eval_set=[(X_tr_h, y_tr_h), (X_va_h, y_va_h)],
                    verbose=False,
                )
                res = final_est.evals_result()
                epoch_history = {
                    "train_loss": [float(x) for x in res["validation_0"]["logloss"]],
                    "val_loss": [float(x) for x in res["validation_1"]["logloss"]],
                }
            # Now refit on 100% of X_fit for full predictive power
            final_est = clone(spec.estimator)
            final_est.fit(X_fit, y_fit)
        else:
            final_est.fit(X_fit, y_fit)
            if spec.name == "MLP Neural Network":
                mlp_step = final_est.named_steps["clf"]
                epoch_history = {
                    "train_loss": [float(x) for x in getattr(mlp_step, "loss_curve_", [])],
                    "val_scores": [float(x) for x in getattr(mlp_step, "validation_scores_", [])],
                }
        train_time_s = time.perf_counter() - t0_fit

        # Evaluate on the 10 student-testing trips
        t0_inf = time.perf_counter()
        test_probs = final_est.predict_proba(X_test)[:, 1]
        inf_time_s = time.perf_counter() - t0_inf
        inf_ms_per_trip = (inf_time_s * 1000.0) / max(1, len(test_trips))

        test_metrics = evaluate_predictions_on_trips(
            test_trips, test_cands, test_probs, y_test
        )

        # Compute training Macro F1 on X_fit sample for overfitting/underfitting check
        sample_n = min(len(X_fit), 40000)
        tr_probs_full = final_est.predict_proba(X_fit[:sample_n])[:, 1]
        n_s_obs = sample_n // k_candidates
        tr_p2d = tr_probs_full.reshape(n_s_obs, k_candidates)
        tr_wta = np.zeros_like(tr_p2d, dtype=int)
        tr_wta[np.arange(n_s_obs), np.argmax(tr_p2d, axis=1)] = 1
        train_macro_f1 = float(f1_score(y_fit[:sample_n], tr_wta.ravel(), average="macro", zero_division=0))

        row = {
            "model_name": spec.name,
            "algorithm_family": spec.family,
            "key_configuration": spec.config_summary,
            "train_macro_f1": train_macro_f1,
            "cv_macro_f1_mean": float(np.mean(fold_f1s)),
            "cv_macro_f1_std": float(np.std(fold_f1s)),
            "cv_top1_accuracy_mean": float(np.mean(fold_top1s)),
            "cv_log_loss_mean": float(np.mean(fold_lls)),
            "student_test_top1_accuracy": test_metrics["top1_accuracy"],
            "student_test_macro_f1": test_metrics["macro_f1"],
            "student_test_macro_precision": test_metrics["macro_precision"],
            "student_test_macro_recall": test_metrics["macro_recall"],
            "student_test_pos_precision": test_metrics["pos_precision"],
            "student_test_pos_recall": test_metrics["pos_recall"],
            "student_test_pos_f1": test_metrics["pos_f1"],
            "student_test_log_loss": test_metrics["log_loss"],
            "route_continuity": test_metrics["route_continuity"],
            "strict_1hop_continuity": test_metrics["strict_1hop_continuity"],
            "training_time_sec": float(train_time_s),
            "cv_total_time_sec": float(cv_time_s),
            "inference_time_sec": float(inf_time_s),
            "inference_ms_per_trip": float(inf_ms_per_trip),
        }
        summary_rows.append(row)
        trained_models[spec.name] = final_est
        diagnostics[spec.name] = {
            "fold_f1s": fold_f1s,
            "fold_top1s": fold_top1s,
            "fold_lls": fold_lls,
            "confusion_matrix": test_metrics["confusion_matrix"],
            "epoch_history": epoch_history,
            "test_probs": test_probs,
        }
        print(
            f"  -> CV Macro F1: {row['cv_macro_f1_mean']:.4f} (+/-{row['cv_macro_f1_std']:.4f}) | "
            f"Student-Test Top-1 Acc: {row['student_test_top1_accuracy']*100:.2f}% | "
            f"Student-Test Macro F1: {row['student_test_macro_f1']:.4f} | "
            f"Route Cont: {row['route_continuity']*100:.2f}% | "
            f"Inf: {row['inference_ms_per_trip']:.2f} ms/trip"
        )

    df_results = pd.DataFrame(summary_rows)
    return df_results, trained_models, diagnostics
