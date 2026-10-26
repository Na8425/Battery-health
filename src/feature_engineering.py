"""
feature_engineering.py — Feature construction without data leakage.

═══════════════════════════════════════════════════════════════════
LEAKAGE ANALYSIS (Dataset: battery_cycle_level_dataset.csv)
═══════════════════════════════════════════════════════════════════

Target variable: soh (State of Health, converted to %)

EXCLUDED as input features — with justification:
──────────────────────────────────────────────────
  • soh          — This IS the target. Using it as input is trivial leakage.

  • capacity     — SOH is defined as:
                     SOH = capacity / initial_capacity
                   Using capacity as an input would allow any model to
                   trivially reconstruct SOH from a simple division.
                   Even without initial_capacity in the dataset, the model
                   would learn a near-perfect linear mapping capacity→SOH.

  • initial_capacity — Reconstructed from the first 3 cycles of each battery.
                   It is a per-battery constant that partially encodes SOH.
                   More importantly, since SOH ≈ capacity / initial_capacity,
                   providing initial_capacity alongside other capacity-related
                   signals would leak the denominator of the SOH formula.

  • rul          — Remaining Useful Life is computed from FUTURE SOH trajectory
                   (number of cycles until SOH < 70 %). At inference time,
                   the true future is unknown; RUL is therefore a label, not
                   a sensor reading, and using it as input leaks future
                   information into the prediction.

SAFE input features:
────────────────────
  • cycle              — Observable: the cycle counter is always known.
  • voltage            — Directly measurable: average discharge voltage (V).
  • temperature        — Directly measurable: cell temperature (°C).
  • battery_id_encoded — Fixed cell identity (label-encoded). Represents
                         chemistry / cell group. Not derived from SOH.

ENGINEERED features (all constructed from safe raw signals only):
─────────────────────────────────────────────────────────────────
  • cycle_norm          — Cycle index normalised to [0,1] per battery.
                          Provides relative age information independent of
                          absolute cycle count.
  • voltage_roll_mean   — 5-cycle rolling mean of voltage per battery.
                          Captures trend rather than point measurement.
  • voltage_roll_std    — 5-cycle rolling std of voltage per battery.
                          Measures voltage instability, which increases
                          with degradation.
  • temp_roll_mean      — 5-cycle rolling mean of temperature per battery.
  • voltage_diff        — Cycle-to-cycle first difference of voltage.
                          Captures rate of voltage change.
  • temp_diff           — Cycle-to-cycle first difference of temperature.
═══════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.utils import get_logger

logger = get_logger("feature_engineering")

FEATURE_COLUMNS = [
    "cycle",
    "voltage",
    "temperature",
    "battery_id_encoded",
    "cycle_norm",
    "voltage_roll_mean",
    "voltage_roll_std",
    "temp_roll_mean",
    "voltage_diff",
    "temp_diff",
]

TARGET_COLUMN = "soh"
ROLL_WINDOW = 5


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame  — cleaned dataset (output of data_loader.clean_dataset)

    Returns
    -------
    pd.DataFrame with additional columns listed in FEATURE_COLUMNS.
    """
    df = df.copy().sort_values(["battery_id", "cycle"]).reset_index(drop=True)

    # ── Label-encode battery_id ───────────────────────────────────────────────
    le = LabelEncoder()
    df["battery_id_encoded"] = le.fit_transform(df["battery_id"])

    # ── Cycle normalised per battery [0, 1] ───────────────────────────────────
    df["cycle_norm"] = df.groupby("battery_id")["cycle"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    )

    # ── Rolling statistics (within each battery) ──────────────────────────────
    grp = df.groupby("battery_id")

    df["voltage_roll_mean"] = grp["voltage"].transform(
        lambda x: x.rolling(ROLL_WINDOW, min_periods=1).mean()
    )
    df["voltage_roll_std"] = grp["voltage"].transform(
        lambda x: x.rolling(ROLL_WINDOW, min_periods=1).std().fillna(0)
    )
    df["temp_roll_mean"] = grp["temperature"].transform(
        lambda x: x.rolling(ROLL_WINDOW, min_periods=1).mean()
    )

    # ── First-order differences (within each battery) ─────────────────────────
    df["voltage_diff"] = grp["voltage"].transform(
        lambda x: x.diff().fillna(0)
    )
    df["temp_diff"] = grp["temperature"].transform(
        lambda x: x.diff().fillna(0)
    )

    logger.info(
        f"Features built. Shape: {df.shape}. "
        f"Input features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}"
    )
    return df


def train_test_split_by_battery(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset at the battery_id level so no test battery
    appears in training data (prevents cross-battery data leakage).

    With 34 batteries and test_fraction=0.20, approximately 7 batteries
    are held out for testing (≈ 277 rows) and 27 are used for training
    (≈ 1138 rows) — consistent with the historical benchmark.

    Parameters
    ----------
    df            : pd.DataFrame
    test_fraction : float  — fraction of batteries to hold out (default 0.20)
    random_state  : int

    Returns
    -------
    (train_df, test_df)
    """
    np.random.seed(random_state)
    battery_ids = np.array(sorted(df["battery_id"].unique()))
    n_test = max(1, int(len(battery_ids) * test_fraction))

    test_ids  = np.random.choice(battery_ids, size=n_test, replace=False)
    train_ids = np.setdiff1d(battery_ids, test_ids)

    train_df = df[df["battery_id"].isin(train_ids)].copy()
    test_df  = df[df["battery_id"].isin(test_ids)].copy()

    logger.info(
        f"Battery-level split: "
        f"{len(train_ids)} train batteries ({len(train_df)} rows) | "
        f"{len(test_ids)} test batteries ({len(test_df)} rows)"
    )
    logger.info(f"  Train IDs: {sorted(train_ids.tolist())}")
    logger.info(f"  Test  IDs: {sorted(test_ids.tolist())}")
    return train_df, test_df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target vector y from a split."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y
