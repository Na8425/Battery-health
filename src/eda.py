"""
eda.py — Exploratory Data Analysis with saved publication-quality figures.

All figures are saved to outputs/figures/ as PNG files.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import FIGURES_DIR, ensure_dirs, get_logger

logger = get_logger("eda")

# ── Style config ──────────────────────────────────────────────────────────────
PALETTE = "tab10"
FIGSIZE_WIDE = (14, 5)
FIGSIZE_SQ   = (10, 8)
FIGSIZE_TALL = (14, 10)
DPI = 150


def _save(fig: plt.Figure, name: str) -> Path:
    ensure_dirs()
    out = FIGURES_DIR / f"{name}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")
    return out


def run_eda(df: pd.DataFrame) -> dict[str, Path]:
    """
    Run all EDA plots and save them.

    Returns
    -------
    dict mapping figure name → saved path
    """
    saved = {}
    saved["soh_vs_cycle"]        = plot_soh_vs_cycle(df)
    saved["capacity_degradation"] = plot_capacity_degradation(df)
    saved["voltage_over_cycles"] = plot_voltage_over_cycles(df)
    saved["temperature_dist"]    = plot_temperature_distribution(df)
    saved["correlation_heatmap"] = plot_correlation_heatmap(df)
    saved["soh_distribution"]    = plot_soh_distribution(df)
    saved["feature_boxplots"]    = plot_feature_boxplots(df)
    logger.info("EDA complete — all figures saved.")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Individual plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_soh_vs_cycle(df: pd.DataFrame) -> Path:
    """SOH degradation trajectory per battery."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    battery_ids = df["battery_id"].unique()
    colors = plt.get_cmap(PALETTE)(np.linspace(0, 1, len(battery_ids)))

    for bid, color in zip(battery_ids, colors):
        sub = df[df["battery_id"] == bid].sort_values("cycle")
        ax.plot(sub["cycle"], sub["soh"], label=f"Battery {bid}",
                color=color, linewidth=1.8, alpha=0.85)

    ax.set_xlabel("Cycle Number", fontsize=13)
    ax.set_ylabel("State of Health (%)", fontsize=13)
    ax.set_title("SOH Degradation per Battery over Cycles", fontsize=15, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(80, color="red", linestyle="--", linewidth=1.2, label="EOL threshold (80%)")
    fig.tight_layout()
    return _save(fig, "soh_vs_cycle")


def plot_capacity_degradation(df: pd.DataFrame) -> Path:
    """Capacity fade per battery."""
    has_init_cap = "initial_capacity" in df.columns
    n_panels = 2 if has_init_cap else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]  # make iterable

    battery_ids = df["battery_id"].unique()
    colors = plt.get_cmap(PALETTE)(np.linspace(0, 1, len(battery_ids)))

    for bid, color in zip(battery_ids, colors):
        sub = df[df["battery_id"] == bid].sort_values("cycle")
        axes[0].plot(sub["cycle"], sub["capacity"],
                     label=f"Battery {bid}", color=color, linewidth=1.8, alpha=0.85)
        if has_init_cap:
            axes[1].plot(sub["cycle"], sub["capacity"] / sub["initial_capacity"] * 100,
                         color=color, linewidth=1.8, alpha=0.85)

    axes[0].set_xlabel("Cycle", fontsize=12)
    axes[0].set_ylabel("Capacity (Ah)", fontsize=12)
    axes[0].set_title("Capacity (Ah) over Cycles", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)

    if has_init_cap:
        axes[1].set_xlabel("Cycle", fontsize=12)
        axes[1].set_ylabel("Retention (%)", fontsize=12)
        axes[1].set_title("Capacity Retention (%) over Cycles", fontsize=13, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, "capacity_degradation")


def plot_voltage_over_cycles(df: pd.DataFrame) -> Path:
    """Voltage trend over cycle life per battery."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    battery_ids = df["battery_id"].unique()
    colors = plt.get_cmap(PALETTE)(np.linspace(0, 1, len(battery_ids)))

    for bid, color in zip(battery_ids, colors):
        sub = df[df["battery_id"] == bid].sort_values("cycle")
        ax.scatter(sub["cycle"], sub["voltage"], label=f"Battery {bid}",
                   color=color, alpha=0.5, s=10)

    ax.set_xlabel("Cycle Number", fontsize=13)
    ax.set_ylabel("Voltage (V)", fontsize=13)
    ax.set_title("Voltage Distribution over Cycle Life", fontsize=15, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "voltage_over_cycles")


def plot_temperature_distribution(df: pd.DataFrame) -> Path:
    """Temperature distribution per battery."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Box plot per battery
    battery_ids = sorted(df["battery_id"].unique())
    data_for_box = [df[df["battery_id"] == bid]["temperature"].dropna().values
                    for bid in battery_ids]
    axes[0].boxplot(data_for_box, labels=battery_ids, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.7))
    axes[0].set_xlabel("Battery ID", fontsize=12)
    axes[0].set_ylabel("Temperature (°C)", fontsize=12)
    axes[0].set_title("Temperature Distribution per Battery", fontsize=13, fontweight="bold")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Temperature vs SOH scatter
    scatter = axes[1].scatter(df["temperature"], df["soh"],
                               c=df["cycle"], cmap="viridis", alpha=0.4, s=15)
    plt.colorbar(scatter, ax=axes[1], label="Cycle")
    axes[1].set_xlabel("Temperature (°C)", fontsize=12)
    axes[1].set_ylabel("SOH (%)", fontsize=12)
    axes[1].set_title("Temperature vs SOH (coloured by Cycle)", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, "temperature_dist")


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    """Feature correlation heatmap."""
    # Only include columns that actually exist in the dataframe
    candidates = ["cycle", "voltage", "temperature", "capacity",
                  "soh", "rul", "initial_capacity"]
    numeric_cols = [c for c in candidates if c in df.columns]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=-1, vmax=1, linewidths=0.5, ax=ax,
        annot_kws={"size": 10}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=15, fontweight="bold", pad=15)
    fig.tight_layout()
    return _save(fig, "correlation_heatmap")


def plot_soh_distribution(df: pd.DataFrame) -> Path:
    """SOH distribution — histogram + KDE per battery."""
    battery_ids = df["battery_id"].unique()
    n = len(battery_ids)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, bid in enumerate(battery_ids):
        sub = df[df["battery_id"] == bid]["soh"].dropna()
        n_bins = min(30, max(1, len(sub) // 3))
        axes[i].hist(sub, bins=n_bins, color="steelblue", edgecolor="white",
                     alpha=0.8, density=True)
        # KDE requires at least 2 unique data points
        if sub.nunique() >= 2:
            try:
                sub.plot.kde(ax=axes[i], color="tomato", linewidth=2)
            except Exception:
                pass  # silently skip KDE if it fails
        axes[i].set_title(f"Battery {bid} — SOH Distribution", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("SOH (%)", fontsize=10)
        axes[i].set_ylabel("Density", fontsize=10)
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("SOH Distributions per Battery", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "soh_distribution")


def plot_feature_boxplots(df: pd.DataFrame) -> Path:
    """Boxplots of key numeric features grouped by battery_id."""
    features = ["voltage", "temperature", "cycle"]
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 5))

    battery_ids = sorted(df["battery_id"].unique())
    colors = plt.get_cmap(PALETTE)(np.linspace(0, 1, len(battery_ids)))

    for ax, feat in zip(axes, features):
        data = [df[df["battery_id"] == bid][feat].dropna().values
                for bid in battery_ids]
        bp = ax.boxplot(data, labels=battery_ids, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(f"{feat.capitalize()} per Battery", fontsize=12, fontweight="bold")
        ax.set_xlabel("Battery ID", fontsize=10)
        ax.set_ylabel(feat.capitalize(), fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return _save(fig, "feature_boxplots")
