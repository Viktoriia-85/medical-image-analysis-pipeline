"""
Patient-Level Feature Aggregation & Validation Module.
Implements the logic to mathematically collapse slice-level texture descriptors 
into a single robust patient-level vector using arithmetic mean aggregation.
Configures StratifiedKFold validation adjusted for patient-level subjects.

This file contains a curated selection of core pipeline fragments and algorithmic logic 
from the original production code. End-to-end file I/O operations and local dataset loops 
are omitted for data privacy and repository cleanliness.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# === EVALUATION CONFIGURATION ===
RANDOM_STATE = 0
N_JOBS = -1
CLASS_WEIGHT_BALANCED = True


def aggregate_slices_to_patients(df: pd.DataFrame,
                                 patient_col: str,
                                 label_col: str,
                                 feat_cols: List[str]) -> pd.DataFrame:
    """
    Collapses slice-level feature tables into a consolidated patient-level space.
    Features are aggregated using the arithmetic mean, while diagnostic labels 
    are mapped using the first occurrence, assuming static per-patient diagnosis.
    """
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for aggregation: {missing[0]}...")

    grouped = df.groupby(patient_col)

    # Compute mean vector per patient across texture signatures
    df_feats = grouped[feat_cols].mean()
    df_labels = grouped[label_col].first()

    df_agg = pd.concat([df_feats, df_labels], axis=1).reset_index()
    return df_agg


def _ensure_label_encoding(y: pd.Series) -> Tuple[np.ndarray, Dict]:
    """Ensures deterministic mapping of text target classes to integer labels."""
    classes = sorted(pd.unique(y))
    mapping = {c: i for i, c in enumerate(classes)}
    y_enc = y.map(mapping).values
    return y_enc, mapping


def _clean_matrix_fit_transform(X_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Cleans the aggregated feature matrix and computes column medians."""
    X = X_df.to_numpy(dtype=float, copy=True)
    X[~np.isfinite(X)] = np.nan
    med = np.nanmedian(X, axis=0)
    nan_all = np.isnan(med)
    med[nan_all] = 0.0
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    return X, med


def _clean_matrix_apply(X_df: pd.DataFrame, med: np.ndarray) -> np.ndarray:
    """Applies pre-computed training medians to clean evaluation/test matrices."""
    X = X_df.to_numpy(dtype=float, copy=True)
    X[~np.isfinite(X)] = np.nan
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    return X


def _build_models_and_grids(n_classes: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Builds baseline estimators and grids optimized for small sample sizes 
    typical of post-aggregation patient cohorts.
    """
    cw = "balanced" if CLASS_WEIGHT_BALANCED else None

    models = {
        "lr": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight=cw, max_iter=15000, random_state=RANDOM_STATE))
        ]),
        "rf": RandomForestClassifier(class_weight=cw, n_jobs=N_JOBS, random_state=RANDOM_STATE),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, class_weight=cw, random_state=RANDOM_STATE))
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier())
        ]),
        "xgb": XGBClassifier(
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            tree_method="hist", random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0
        ),
        "lgbm": LGBMClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1)
    }

    grids = {
        "lr": [
            {"clf__penalty": ["elasticnet"], "clf__solver": ["saga"],
             "clf__l1_ratio": [0.1, 0.5, 0.9], "clf__C": [0.05, 0.1, 0.5]},
            {"clf__penalty": ["l2"], "clf__solver": ["lbfgs"], "clf__C": [0.05, 0.1, 0.5]},
        ],
        "rf": {
            "n_estimators": [25, 50, 100],
            "max_depth": [3, 4, 6],
            "min_samples_leaf": [5, 7],
        },
        "svm": {
            "clf__kernel": ["linear", "rbf"],
            "clf__C": [0.1, 0.5, 1.0, 2.0],
            "clf__gamma": ["scale", "auto"],
            "clf__decision_function_shape": ["ovr", "ovo"],
        },
        "knn": {
            "clf__n_neighbors": [7, 9, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [2],
        },
        "xgb": {
            "n_estimators": [15, 20, 30],
            "max_depth": [2, 3],
            "learning_rate": [0.01, 0.03, 0.05],
            "reg_alpha": [0.1, 1.0, 5.0, 7.0],
            "reg_lambda": [1.0, 5.0, 10.0],
            "subsample": [0.6, 0.8, 1.0],
        },
        "lgbm": {
            "n_estimators": [15, 20, 25],
            "num_leaves": [2, 3, 4],
            "min_child_samples": [5, 7, 10, 15],
            "learning_rate": [0.01, 0.03, 0.05],
            "reg_alpha": [1.0, 3.0, 5.0],
            "reg_lambda": [1.0, 3.0, 5.0, 7.0],
        },
    }
    return models, grids
