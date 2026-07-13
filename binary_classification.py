"""
Binary Classification & Anti-Overfitting Optimization Module.
Implements binary classification (VLS vs non-VLS) among COVID-19 cohorts.
Features a custom training pipeline with internal GroupShuffleSplit for Early Stopping 
in gradient boosting architectures (XGBoost/LightGBM) to prevent validation leakage.

This file contains a curated selection of core pipeline fragments and algorithmic logic 
from the original production code. End-to-end file I/O operations and local dataset loops 
are omitted for data privacy and repository cleanliness.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_random_state
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# === EVALUATION CONFIGURATION ===
RANDOM_STATE = 0
N_JOBS = -1
CLASS_WEIGHT_BALANCED = True


def _build_models_and_grids(n_classes: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Builds baseline estimators and conservative parameter grids explicitly 
    configured with higher regularization to mitigate overfitting on thin clinical subsets.
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
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0
        ),
        "lgbm": LGBMClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1, class_weight='balanced')
    }

    grids = {
        "lr": [
            {"clf__penalty": ["l2"], "clf__solver": ["lbfgs", "newton-cg"], "clf__C": [0.01, 0.1, 1.0, 10.0]},
            {"clf__penalty": ["l1"], "clf__solver": ["saga"], "clf__C": [0.01, 0.1, 1.0, 10.0]},
            {"clf__penalty": ["elasticnet"], "clf__solver": ["saga"], "clf__l1_ratio": [0.2, 0.5, 0.8], "clf__C": [0.01, 0.1, 1.0, 10.0]},
        ],
        "rf": {
            "n_estimators": [200, 300],
            "max_depth": [4, 6, 8],
            "min_samples_leaf": [5, 10, 20],
            "min_samples_split": [10, 20, 50],
            "max_features": ["sqrt", "log2"],
        },
        "svm": {
            "clf__kernel": ["rbf"],
            "clf__C": [0.1, 1.0, 10.0],
            "clf__gamma": ["scale", "auto"],
        },
        "knn": {
            "clf__n_neighbors": [5, 9, 15, 21],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [2],
        },
        "xgb": {
            "n_estimators": [100, 300, 600],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.7, 0.85],
            "colsample_bytree": [0.7, 0.9],
            "min_child_weight": [1, 3],
            "gamma": [0.0, 0.5],
            "reg_lambda": [1.0, 2.0],
            "reg_alpha": [0.0, 0.5],
        },
        "lgbm": {
            "n_estimators": [100, 300, 600],
            "num_leaves": [31, 63],
            "min_child_samples": [20, 40],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9],
            "reg_lambda": [0.0, 1.0],
            "reg_alpha": [0.0, 0.5],
        },
    }
    return models, grids


def _refit_with_early_stopping(pipeline_or_est: BaseEstimator,
                               key: str,
                               X_tv: np.ndarray, y_tv: np.ndarray,
                               groups_tv: np.ndarray,
                               random_state: int = 0) -> BaseEstimator:
    """
    Executes advanced training stabilization for gradient boosted trees (XGB/LGBM).
    Extracts a 15% out-of-fold validation block strictly partitioned by groups 
    to track optimal tree generation. Then re-initializes and refits the model on 
    the full set up to the tracked 'best_iteration' threshold.
    """
    est = clone(pipeline_or_est)
    clf = est.named_steps["clf"] if hasattr(est, "named_steps") and "clf" in est.named_steps else est

    if not isinstance(clf, (XGBClassifier, LGBMClassifier)):
        est.fit(X_tv, y_tv)
        return est

    rng = check_random_state(random_state)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=rng)
    tr_idx, es_idx = next(gss.split(X_tv, y_tv, groups_tv))

    X_tr_es, y_tr_es = X_tv[tr_idx], y_tv[tr_idx]
    X_val_es, y_val_es = X_tv[es_idx], y_tv[es_idx]

    es_est = clone(est)

    if isinstance(clf, XGBClassifier):
        es_est.set_params(**{"clf__n_estimators": 2000} if hasattr(es_est, "named_steps") else {"n_estimators": 2000})
        fit_params = {
            "clf__eval_set": [(X_val_es, y_val_es)] if hasattr(es_est, "named_steps") else None,
            "clf__early_stopping_rounds": 50 if hasattr(es_est, "named_steps") else None,
            "clf__verbose": False if hasattr(es_est, "named_steps") else None
        }
    else:  # LGBM
        es_est.set_params(**{"clf__n_estimators": 2000} if hasattr(es_est, "named_steps") else {"n_estimators": 2000})
        fit_params = {
            "clf__eval_set": [(X_val_es, y_val_es)] if hasattr(es_est, "named_steps") else None,
            "clf__early_stopping_rounds": 50 if hasattr(es_est, "named_steps") else None,
            "clf__verbose": -1 if hasattr(es_est, "named_steps") else None
        }

    clean_fit_params = {k: v for k, v in fit_params.items() if v is not None}
    es_est.fit(X_tr_es, y_tr_es, **clean_fit_params)

    best_n = None
    clf_es = es_est.named_steps["clf"] if hasattr(es_est, "named_steps") and "clf" in es_est.named_steps else es_est
    if isinstance(clf_es, XGBClassifier) and hasattr(clf_es, "best_iteration"):
        best_n = int(clf_es.best_iteration) if clf_es.best_iteration is not None else None
    if isinstance(clf_es, LGBMClassifier) and hasattr(clf_es, "best_iteration_"):
        best_n = int(clf_es.best_iteration_) if clf_es.best_iteration_ is not None else None

    if not best_n or best_n <= 0:
        return es_est

    final_est = clone(est)
    if hasattr(final_est, "named_steps") and "clf" in final_est.named_steps:
        final_est.set_params(**{"clf__n_estimators": best_n})
    else:
        final_est.set_params(**{"n_estimators": best_n})
        
    final_est.fit(X_tv, y_tv)
    return final_est
