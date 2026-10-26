"""
evaluate.py — Model evaluation for EV Battery SOH Prediction.

Metrics: MAE, RMSE, R²
Produces a comparison table and a residuals/prediction plot.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils import FIGURES_DIR, ensure_dirs, save_metrics, get_logger

logger = get_logger("evaluate")

DPI = 150


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict:
    """
    Compute MAE, RMSE, and R² for a single model on test data.

    Parameters
    ----------
    model      : fitted sklearn-compatible model
    X_test     : test feature matrix
    y_test     : true SOH values
    model_name : label for logging

    Returns
    -------
    dict with keys: model, mae, rmse, r2, n_test, predictions
    """
    y_pred = model.predict(X_test)

    mae  = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))

    logger.info(
        f"[{model_name}] MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  "
        f"(n_test={len(y_test)})"
    )

    return {
        "model":       model_name,
        "mae":         mae,
        "rmse":        rmse,
        "r2":          r2,
        "n_test":      int(len(y_test)),
        "predictions": y_pred.tolist(),
    }


def evaluate_all(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate all models and return a nested dict of results.

    Parameters
    ----------
    models : {'model_name': fitted_model, ...}
    X_test, y_test : test split

    Returns
    -------
    dict keyed by model name, each value is a metrics dict.
    """
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test, name)

    # ── Print comparison table ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'N Test':>8}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20} {r['mae']:>8.4f} {r['rmse']:>8.4f} "
              f"{r['r2']:>8.4f} {r['n_test']:>8}")
    print("=" * 60 + "\n")

    save_metrics(results)
    return results


def plot_predictions(
    results: dict,
    y_test: pd.Series,
    test_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    """
    Generate and save prediction plots for all evaluated models.

    Plots saved:
    - pred_vs_actual_<model>.png  — scatter of predicted vs actual SOH
    - residuals_<model>.png       — residuals over test samples

    Returns
    -------
    dict mapping plot name → saved path (as str)
    """
    ensure_dirs()
    saved = {}
    y_true = np.array(y_test)

    for name, r in results.items():
        y_pred = np.array(r["predictions"])

        # ── Predicted vs Actual ────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=18, color="steelblue",
                        edgecolors="none")
        lim = [min(y_true.min(), y_pred.min()) - 2,
               max(y_true.max(), y_pred.max()) + 2]
        axes[0].plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
        axes[0].set_xlabel("Actual SOH (%)", fontsize=12)
        axes[0].set_ylabel("Predicted SOH (%)", fontsize=12)
        axes[0].set_title(f"{name} — Predicted vs Actual SOH", fontsize=13,
                          fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].text(0.05, 0.92,
                     f"MAE={r['mae']:.4f}\nRMSE={r['rmse']:.4f}\nR²={r['r2']:.4f}",
                     transform=axes[0].transAxes, fontsize=10,
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        # Residuals
        residuals = y_pred - y_true
        axes[1].scatter(range(len(residuals)), residuals, alpha=0.5, s=15,
                        color="tomato", edgecolors="none")
        axes[1].axhline(0, color="black", linewidth=1.2, linestyle="--")
        axes[1].set_xlabel("Test Sample Index", fontsize=12)
        axes[1].set_ylabel("Residual (Predicted − Actual)", fontsize=12)
        axes[1].set_title(f"{name} — Residuals", fontsize=13, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        path = FIGURES_DIR / f"predictions_{name}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        saved[f"predictions_{name}"] = str(path)
        logger.info(f"Saved: {path}")

    return saved


def plot_model_comparison(results: dict) -> str:
    """Bar chart comparing MAE, RMSE, R² across all models."""
    ensure_dirs()
    names  = list(results.keys())
    maes   = [results[n]["mae"]  for n in names]
    rmses  = [results[n]["rmse"] for n in names]
    r2s    = [results[n]["r2"]   for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, vals, label, color in zip(
        axes,
        [maes, rmses, r2s],
        ["MAE (lower is better)", "RMSE (lower is better)", "R² (higher is better)"],
        ["steelblue", "tomato", "mediumseagreen"],
    ):
        bars = ax.bar(x, vals, color=color, alpha=0.8, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=10)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002 * max(vals),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Model Comparison — SOH Prediction", fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = str(FIGURES_DIR / "model_comparison.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_feature_importance(model, feature_names: list, model_name: str) -> str | None:
    """
    Plot feature importance for tree-based models (RF, XGBoost).
    Returns path to saved figure, or None if model doesn't support it.
    """
    ensure_dirs()
    # Extract importance from RF or XGBoost
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        # Pipeline — try last estimator
        last_step = list(model.named_steps.values())[-1]
        if hasattr(last_step, "feature_importances_"):
            importances = last_step.feature_importances_
        else:
            return None
    else:
        return None

    # Sort
    idx  = np.argsort(importances)[::-1]
    importance_sorted = importances[idx]
    names_sorted = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names_sorted[::-1], importance_sorted[::-1],
                   color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(f"{model_name} — Feature Importance", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = str(FIGURES_DIR / f"feature_importance_{model_name}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path
