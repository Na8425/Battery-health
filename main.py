"""
main.py — End-to-end pipeline runner for EV Battery Health Prediction.

Run:
    python main.py

This script:
  1. Loads and inspects the dataset
  2. Runs exploratory data analysis (saves figures)
  3. Engineers features (no data leakage)
  4. Splits by battery_id (train / test)
  5. Trains Random Forest, XGBoost, and MLP
  6. Evaluates all models and saves metrics
  7. Generates prediction and comparison plots
"""

import sys
from pathlib import Path

# ── Ensure project root is on PYTHONPATH ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_dataset, inspect_dataset, clean_dataset
from src.eda import run_eda
from src.feature_engineering import (
    build_features,
    train_test_split_by_battery,
    get_X_y,
    FEATURE_COLUMNS,
)
from src.train import train_all
from src.evaluate import (
    evaluate_all,
    plot_predictions,
    plot_model_comparison,
    plot_feature_importance,
)
from src.utils import ensure_dirs, get_logger

logger = get_logger("main")


def run_pipeline():
    ensure_dirs()
    print("\n" + "█" * 60)
    print("  EV Battery Health Prediction — ML Pipeline")
    print("█" * 60 + "\n")

    # ── Step 1: Load & Inspect ────────────────────────────────────────────────
    print("Step 1: Loading and inspecting dataset…")
    df_raw = load_dataset()
    report = inspect_dataset(df_raw)

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    print("\nStep 2: Cleaning dataset…")
    df_clean = clean_dataset(df_raw)

    # ── Step 3: EDA ───────────────────────────────────────────────────────────
    print("\nStep 3: Running exploratory data analysis…")
    eda_paths = run_eda(df_clean)
    print(f"  Saved {len(eda_paths)} EDA figures to outputs/figures/")

    # ── Step 4: Feature Engineering ───────────────────────────────────────────
    print("\nStep 4: Engineering features (no leakage)…")
    df_feat = build_features(df_clean)

    # ── Step 5: Train/Test Split ──────────────────────────────────────────────
    print("\nStep 5: Splitting by battery_id…")
    train_df, test_df = train_test_split_by_battery(df_feat, test_fraction=0.2)
    X_train, y_train = get_X_y(train_df)
    X_test,  y_test  = get_X_y(test_df)

    print(f"  Training: {X_train.shape[0]} samples | "
          f"Testing: {X_test.shape[0]} samples")

    # ── Step 6: Train All Models ──────────────────────────────────────────────
    print("\nStep 6: Training models…")
    models = train_all(X_train, y_train)

    # ── Step 7: Evaluate ──────────────────────────────────────────────────────
    print("\nStep 7: Evaluating models…")
    results = evaluate_all(models, X_test, y_test)

    # ── Step 8: Generate Plots ────────────────────────────────────────────────
    print("\nStep 8: Generating evaluation plots…")
    plot_predictions(results, y_test, test_df)
    plot_model_comparison(results)

    for model_name, model in models.items():
        plot_feature_importance(model, FEATURE_COLUMNS, model_name)

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  Pipeline complete! Results saved to outputs/")
    print("  Run:  streamlit run app.py  to launch the dashboard")
    print("█" * 60 + "\n")

    return results


if __name__ == "__main__":
    run_pipeline()
