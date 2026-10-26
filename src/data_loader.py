"""
data_loader.py — Dataset loading, validation, and inspection.

Real Dataset: battery_cycle_level_dataset.csv
Columns: battery_id, cycle, voltage, temperature, capacity, soh, rul

NOTE on SOH:
  In this dataset soh is expressed as a ratio (e.g. 1.000 = 100 %, 0.80 = 80 %).
  We convert it to percentage (×100) immediately after loading so all downstream
  code works in the conventional 0–100 % scale.

NOTE on initial_capacity:
  The column is absent from this dataset.  We reconstruct it per battery as the
  mean capacity of the first 3 cycles (a standard approximation used in the
  battery-health literature).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import DATA_DIR, get_logger

logger = get_logger("data_loader")

REQUIRED_COLUMNS = [
    "battery_id",
    "cycle",
    "voltage",
    "temperature",
    "capacity",
    "soh",
    "rul",
]


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """
    Load the battery dataset from CSV.

    Parameters
    ----------
    path : Path, optional
        Override the default data directory path.

    Returns
    -------
    pd.DataFrame
        Raw dataset with correct dtypes applied and soh converted to %.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the expected location.
    ValueError
        If required columns are missing from the file.
    """
    if path is None:
        path = DATA_DIR / "battery_dataset.csv"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please place battery_dataset.csv in the data/ folder.\n"
            "Source: battery_cycle_level_dataset.csv from Downloads."
        )

    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw shape: {df.shape}")

    _validate_columns(df)
    df = _cast_types(df)
    df = _normalise_soh(df)
    df = _reconstruct_initial_capacity(df)
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
    logger.info("Column validation passed.")


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct dtypes for all columns."""
    df["battery_id"] = df["battery_id"].astype(str)
    for col in ["cycle", "voltage", "temperature", "capacity", "soh", "rul"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalise_soh(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert SOH from ratio [0–1] to percentage [0–100] if needed.
    Detects the scale automatically: if max SOH ≤ 2.0 it is treated as a ratio.
    """
    if df["soh"].max() <= 2.0:
        logger.info(f"SOH detected as ratio (max={df['soh'].max():.4f}). "
                    "Converting to percentage (×100).")
        df["soh"] = df["soh"] * 100.0
    else:
        logger.info(f"SOH already in percentage scale (max={df['soh'].max():.2f}%).")
    return df


def _reconstruct_initial_capacity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct initial_capacity per battery as the mean capacity of the
    first 3 discharge cycles (standard approximation).
    Adds column 'initial_capacity' to the dataframe.
    """
    df = df.sort_values(["battery_id", "cycle"]).copy()
    init_cap = (
        df.groupby("battery_id")
        .apply(lambda g: g.nsmallest(3, "cycle")["capacity"].mean(), include_groups=False)
        .rename("initial_capacity")
    )
    df = df.merge(init_cap, on="battery_id", how="left")
    logger.info("Reconstructed initial_capacity from first-3-cycle mean per battery.")
    return df


# ── Inspection report ─────────────────────────────────────────────────────────

def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Run a full data quality inspection and return a summary dictionary.
    """
    report = {}

    report["shape"] = df.shape
    report["dtypes"] = df.dtypes.astype(str).to_dict()

    # Missing values
    missing = df.isnull().sum()
    report["missing_values"] = missing[missing > 0].to_dict()
    total_missing = int(missing.sum())
    report["total_missing_cells"] = total_missing

    # Duplicates
    n_dupes = int(df.duplicated().sum())
    report["duplicate_rows"] = n_dupes

    # Battery IDs
    report["battery_ids"] = sorted(df["battery_id"].unique().tolist())
    report["n_batteries"] = int(df["battery_id"].nunique())

    # Cycle range
    report["cycle_range"] = {
        "min": int(df["cycle"].min()),
        "max": int(df["cycle"].max()),
    }

    # SOH range (in %)
    report["soh_range"] = {
        "min": float(df["soh"].min()),
        "max": float(df["soh"].max()),
        "mean": float(df["soh"].mean()),
    }

    # Numeric summary
    report["numeric_summary"] = df.describe().round(4).to_dict()

    # ── Console print ──────────────────────────────────────────────────────────
    print("=" * 65)
    print("DATASET INSPECTION REPORT")
    print("=" * 65)
    print(f"  Shape              : {report['shape'][0]} rows × {report['shape'][1]} cols")
    print(f"  Unique batteries   : {report['n_batteries']}")
    print(f"  Battery IDs        : {report['battery_ids']}")
    print(f"  Cycle range        : {report['cycle_range']['min']} – {report['cycle_range']['max']}")
    print(f"  SOH range          : {report['soh_range']['min']:.2f}% – {report['soh_range']['max']:.2f}%")
    print(f"  SOH mean           : {report['soh_range']['mean']:.2f}%")
    print(f"  Missing values     : {total_missing} cells")
    if report["missing_values"]:
        for col, cnt in report["missing_values"].items():
            print(f"    └─ {col}: {cnt} missing")
    print(f"  Duplicate rows     : {n_dupes}")
    print()
    print("Data Types:")
    for col, dtype in report["dtypes"].items():
        print(f"  {col:<22} {dtype}")
    print("=" * 65)

    logger.info(f"Inspection complete: {report['shape']}, {report['n_batteries']} batteries, "
                f"{total_missing} missing, {n_dupes} duplicates")
    return report


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Drop duplicate rows
    - Drop rows where target (soh) is NaN
    - Fill remaining numeric NaNs with column median per battery

    Returns
    -------
    pd.DataFrame  — cleaned copy
    """
    original_len = len(df)
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=["soh"])

    numeric_cols = ["voltage", "temperature", "capacity", "rul", "initial_capacity"]
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df.groupby("battery_id")[col].transform(
                lambda x: x.fillna(x.median())
            )

    logger.info(f"Cleaning: {original_len} → {len(df)} rows "
                f"(removed {original_len - len(df)})")
    return df
