"""
generate_dataset.py — Generate a realistic synthetic battery dataset
modelled on the NASA Battery Prognostics Dataset (PCoE).

The NASA dataset contains discharge cycles from Li-ion batteries (B0005,
B0006, B0007, B0018) tested at various temperatures. SOH is defined as:
    SOH (%) = (capacity / initial_capacity) * 100

This generator produces per-cycle measurements that faithfully reproduce:
  - Non-linear capacity fade (accelerating near end-of-life)
  - Voltage decrease correlated with cycle count
  - Temperature variation with mild cycle correlation
  - Battery-to-battery variability in initial capacity and fade rate
  - Noise typical of real charge/discharge measurements

Run:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

# ── Battery definitions (inspired by NASA B0005–B0018) ────────────────────────
BATTERIES = {
    "B0005": {"initial_capacity": 1.856, "fade_rate": 0.00055, "temp_mean": 24.0},
    "B0006": {"initial_capacity": 1.863, "fade_rate": 0.00050, "temp_mean": 24.0},
    "B0007": {"initial_capacity": 1.894, "fade_rate": 0.00048, "temp_mean": 24.0},
    "B0018": {"initial_capacity": 1.860, "fade_rate": 0.00058, "temp_mean": 40.0},
    "B0025": {"initial_capacity": 1.852, "fade_rate": 0.00052, "temp_mean": 30.0},
    "B0029": {"initial_capacity": 1.845, "fade_rate": 0.00060, "temp_mean": 35.0},
    "B0032": {"initial_capacity": 1.870, "fade_rate": 0.00046, "temp_mean": 20.0},
    "B0034": {"initial_capacity": 1.838, "fade_rate": 0.00063, "temp_mean": 45.0},
}

EOL_SOH = 70.0   # End of life threshold
MAX_CYCLES = 500


def fade_model(cycle: np.ndarray, initial: float, rate: float) -> np.ndarray:
    """
    Non-linear capacity fade model.
    Uses a combination of linear and exponential decay.
    """
    linear_fade = rate * cycle
    # Accelerated fade at later cycles
    accel_fade  = 0.0002 * np.exp(0.005 * cycle) - 0.0002
    capacity    = initial * (1 - linear_fade - accel_fade)
    # Add realistic measurement noise (σ ≈ 3 mAh)
    noise       = rng.normal(0, 0.003, size=len(cycle))
    return np.clip(capacity + noise, 0.5, initial * 1.02)


def voltage_model(cycle: np.ndarray, capacity_ratio: np.ndarray) -> np.ndarray:
    """
    Voltage during discharge (average, V).
    Typically ~3.6–4.2 V for Li-ion; decreases with degradation.
    """
    base_voltage = 3.75 + 0.30 * capacity_ratio        # higher SOH → higher V
    noise        = rng.normal(0, 0.02, size=len(cycle))
    return np.clip(base_voltage + noise, 2.5, 4.2)


def temperature_model(cycle: np.ndarray, temp_mean: float) -> np.ndarray:
    """
    Cell temperature during discharge (°C).
    Increases slightly with cycle count as internal resistance grows.
    """
    drift = 0.002 * cycle                               # gentle thermal drift
    noise = rng.normal(0, 0.8, size=len(cycle))
    return temp_mean + drift + noise


def generate() -> pd.DataFrame:
    rows = []
    for battery_id, params in BATTERIES.items():
        initial_cap = params["initial_capacity"]
        fade_rate   = params["fade_rate"]
        temp_mean   = params["temp_mean"]

        cycles = np.arange(1, MAX_CYCLES + 1)
        capacity    = fade_model(cycles, initial_cap, fade_rate)
        soh         = (capacity / initial_cap) * 100.0
        voltage     = voltage_model(cycles, capacity / initial_cap)
        temperature = temperature_model(cycles, temp_mean)

        # Find true EOL cycle
        eol_cycle = MAX_CYCLES
        below_eol = np.where(soh < EOL_SOH)[0]
        if len(below_eol) > 0:
            eol_cycle = int(below_eol[0]) + 1   # first cycle below EOL

        rul = np.maximum(eol_cycle - cycles, 0).astype(float)
        # Add noise to RUL (real RUL estimation has uncertainty)
        rul += rng.normal(0, 2, size=len(cycles))
        rul = np.maximum(rul, 0)

        for i, c in enumerate(cycles):
            rows.append({
                "battery_id":       battery_id,
                "cycle":            int(c),
                "voltage":          round(float(voltage[i]), 4),
                "temperature":      round(float(temperature[i]), 4),
                "capacity":         round(float(capacity[i]), 6),
                "soh":              round(float(soh[i]), 4),
                "rul":              round(float(rul[i]), 2),
                "initial_capacity": round(float(initial_cap), 6),
            })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "battery_dataset.csv"

    df = generate()
    df.to_csv(out_path, index=False)

    print(f"Dataset generated: {out_path}")
    print(f"  Shape            : {df.shape}")
    print(f"  Batteries        : {sorted(df['battery_id'].unique())}")
    print(f"  Cycles per batt  : {df.groupby('battery_id')['cycle'].count().to_dict()}")
    print(f"  SOH range        : {df['soh'].min():.2f}% – {df['soh'].max():.2f}%")
    print(f"  Capacity range   : {df['capacity'].min():.4f} – {df['capacity'].max():.4f} Ah")
