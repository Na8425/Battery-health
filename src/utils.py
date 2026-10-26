"""
utils.py — Shared utility helpers for EV Battery Health Prediction project.
"""

import os
import json
import logging
from pathlib import Path

# ── Project root (one level above src/) ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR    = PROJECT_ROOT / "data"
MODELS_DIR  = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_FILE = OUTPUTS_DIR / "metrics.json"

# ── Logging setup ─────────────────────────────────────────────────────────────
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


# ── Directory helpers ─────────────────────────────────────────────────────────
def ensure_dirs() -> None:
    """Create all required output directories if they don't exist."""
    for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ── Metrics persistence ───────────────────────────────────────────────────────
def save_metrics(metrics: dict) -> None:
    """Save evaluation metrics dictionary to outputs/metrics.json."""
    ensure_dirs()
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    get_logger("utils").info(f"Metrics saved to {METRICS_FILE}")


def load_metrics() -> dict:
    """Load metrics from outputs/metrics.json, return empty dict if missing."""
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return {}
