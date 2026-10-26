"""
train.py — Model training for EV Battery SOH Prediction.

Models
------
1. Random Forest Regressor  (scikit-learn)
2. XGBoost Regressor        (xgboost)
3. MLP Neural Network       (scikit-learn MLPRegressor)

All models are saved to models/ as .joblib files.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.utils import MODELS_DIR, ensure_dirs, get_logger

logger = get_logger("train")


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
    save: bool = True,
) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor.

    Parameters
    ----------
    X_train, y_train : training features and labels
    n_estimators     : number of trees
    max_depth        : maximum tree depth (None = unlimited)
    random_state     : reproducibility seed
    save             : save model to models/ if True

    Returns
    -------
    Trained RandomForestRegressor
    """
    logger.info(f"Training Random Forest (n_estimators={n_estimators}, "
                f"max_depth={max_depth})...")
    logger.info(f"  Training set: {X_train.shape[0]} rows × {X_train.shape[1]} features")

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    logger.info("Random Forest training complete.")

    if save:
        _save_model(rf, "random_forest")

    return rf


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    random_state: int = 42,
    save: bool = True,
) -> xgb.XGBRegressor:
    """
    Train an XGBoost Regressor.

    Parameters
    ----------
    X_train, y_train : training data
    n_estimators     : number of boosting rounds
    learning_rate    : step size shrinkage
    max_depth        : max depth of each tree
    random_state     : seed

    Returns
    -------
    Trained XGBRegressor
    """
    logger.info(f"Training XGBoost (n_estimators={n_estimators}, "
                f"lr={learning_rate}, max_depth={max_depth})...")

    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    logger.info("XGBoost training complete.")

    if save:
        _save_model(xgb_model, "xgboost")

    return xgb_model


def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hidden_layers: tuple = (128, 64, 32),
    max_iter: int = 500,
    random_state: int = 42,
    save: bool = True,
) -> Pipeline:
    """
    Train an MLP Neural Network wrapped in a StandardScaler Pipeline.

    MLPs are sensitive to feature scale, so we wrap in a Pipeline that
    applies StandardScaler before the network.

    Parameters
    ----------
    X_train, y_train : training data
    hidden_layers    : tuple of hidden layer sizes
    max_iter         : maximum training epochs
    random_state     : seed

    Returns
    -------
    sklearn Pipeline (scaler + MLPRegressor)
    """
    logger.info(f"Training MLP (layers={hidden_layers}, max_iter={max_iter})...")

    mlp_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
        )),
    ])
    mlp_pipeline.fit(X_train, y_train)
    logger.info("MLP training complete.")

    if save:
        _save_model(mlp_pipeline, "mlp")

    return mlp_pipeline


# ── Model persistence ─────────────────────────────────────────────────────────

def _save_model(model, name: str) -> Path:
    """Persist a model to models/<name>.joblib."""
    ensure_dirs()
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    logger.info(f"Model saved: {path}")
    return path


def load_model(name: str):
    """Load a previously saved model by name (without .joblib extension)."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded: {path}")
    return model


def train_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """
    Train all three models and return them in a dict.

    Returns
    -------
    {'random_forest': ..., 'xgboost': ..., 'mlp': ...}
    """
    logger.info("=== Training all models ===")
    models = {}
    models["random_forest"] = train_random_forest(X_train, y_train)
    models["xgboost"]       = train_xgboost(X_train, y_train)
    models["mlp"]           = train_mlp(X_train, y_train)
    logger.info("=== All models trained ===")
    return models
