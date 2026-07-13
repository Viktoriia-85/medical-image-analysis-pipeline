"""
Multi-class Classification & Hyperparameter Optimization Module.
Configures automated search (GridSearchCV) for multiple families of algorithms 
(Logistic Regression, RF, SVM, KNN, XGBoost, LightGBM) on slice-level features.

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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# === EVALUATION CONFIGURATION ===
RANDOM_STATE = 0
N_JOBS = -1
CLASS_WEIGHT_BALANCED = True

FEATURE_PREFIXES_INCLUDE = ("GLCM_", "GLDS_", "GLRLM_", "LBP_")
FEATURE_PREFIXES_EXCLUDE = ("__warn_",)


def _select_features(df: pd.DataFrame) -> List[str]:
    """Selects numeric texture feature columns based on specific inclusions/exclusions."""
    cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in FEATURE_PREFIXES_EXCLUDE):
            continue
        if any(c.startswith(p) for p in FEATURE_PREFIXES_INCLUDE) and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _ensure_label_encoding(y: pd.Series) -> Tuple[np.ndarray, Dict]:
    """Ensures deterministic mapping of text target classes to integer labels."""
    classes = sorted(pd.unique(y))
    mapping = {c: i for i, c in enumerate(classes)}
    y_enc = y.map(mapping).values
    return y_enc, mapping


def _clean_matrix_fit_transform(X_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Cleans the feature matrix, computes column medians, and imputes missing/infinite values."""
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
    Builds a dictionary of scikit-learn compatible classifiers and 
    their respective parameter grids tailored for texture spaces.
    """
    cw = "balanced" if CLASS_WEIGHT_BALANCED else None

    models = {
        "lr": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight=cw, max_iter=5000, random_state=RANDOM_STATE))
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
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 4, 6],
            "min_samples_leaf": [5, 7],
        },
        "svm": {
            "clf__kernel": ["linear", "rbf"],
            "clf__C": [0.1, 0.5, 1.0, 2.0],
            "clf__gamma": ["scale", "auto"],
        },
        "knn": {
            "clf__n_neighbors": [3, 5, 7],
            "clf__weights": ["distance"],
            "clf__p": [1, 2],
        },
        "xgb": {
            "n_estimators": [15, 20, 25, 50],
            "max_depth": [2, 3],
            "learning_rate": [0.01, 0.03, 0.05],
            "reg_alpha": [0.1, 1.0, 5.0, 7.0],   # L1 regularization for feature pruning
            "reg_lambda": [1.0, 5.0, 10.0],     # L2 regularization for weight smoothing
            "subsample": [0.6, 0.8, 1.0],
        },
        "lgbm": {
            "n_estimators": [15, 20, 30, 50],
            "num_leaves": [2, 3, 4],
            "min_child_samples": [5, 7, 10, 15],
            "learning_rate": [0.01, 0.03, 0.05],
            "reg_alpha": [1.0, 3.0, 5.0],
            "reg_lambda": [1.0, 3.0, 5.0, 7.0],
        },
    }
    return models, grids
