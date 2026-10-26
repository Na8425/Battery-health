"""
app.py — Streamlit dashboard for EV Battery Health Prediction.

Features:
  - Battery selector with degradation curve
  - Real-time SOH prediction input
  - Model comparison table
  - Feature importance charts
  - EDA figure gallery

Run:  streamlit run app.py
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# ── Ensure project root on path ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import (
    DATA_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR,
    METRICS_FILE, load_metrics, ensure_dirs,
)
from src.data_loader import load_dataset, clean_dataset
from src.feature_engineering import (
    build_features, train_test_split_by_battery,
    get_X_y, FEATURE_COLUMNS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EV Battery Health Prediction",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.main-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}

.main-header p {
    font-size: 1.0rem;
    opacity: 0.85;
    margin: 0.5rem 0 0 0;
    font-weight: 300;
}

.metric-card {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: white;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    transition: transform 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
}

.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
}

.metric-card .label {
    font-size: 0.85rem;
    opacity: 0.8;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.soh-gauge {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
}

.badge-good { color: #2ecc71; }
.badge-warn { color: #f39c12; }
.badge-crit { color: #e74c3c; }

.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #2c5364;
    border-left: 4px solid #2a5298;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

.model-table th { background: #2a5298 !important; color: white !important; }

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    """Load, clean, and feature-engineer the dataset."""
    try:
        df = load_dataset()
        df = clean_dataset(df)
        df = build_features(df)
        return df, None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_trained_models():
    """Load all saved .joblib models from models/."""
    import joblib
    loaded = {}
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        name = path.stem
        try:
            loaded[name] = joblib.load(path)
        except Exception as e:
            st.warning(f"Could not load model '{name}': {e}")
    return loaded


def soh_color_class(soh: float) -> str:
    if soh >= 85:
        return "badge-good"
    elif soh >= 70:
        return "badge-warn"
    else:
        return "badge-crit"


def soh_status(soh: float) -> str:
    if soh >= 85:
        return "✅ Healthy"
    elif soh >= 70:
        return "⚠️ Degraded"
    else:
        return "🔴 End of Life"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ensure_dirs()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🔋 EV Battery Health Prediction</h1>
        <p>Advanced Machine Learning — NASA Battery Cycle-Level Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/battery.png", width=60)
        st.markdown("### Navigation")
        page = st.radio(
            "Select section",
            ["📊 Overview", "🔍 EDA Plots", "🤖 Model Results",
             "🎯 SOH Predictor", "📈 Feature Importance"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("**Dataset**")
        dataset_path = DATA_DIR / "battery_dataset.csv"
        if dataset_path.exists():
            st.success(f"✅ Dataset found")
        else:
            st.error("❌ Dataset not found")
            st.info("Run `python generate_dataset.py` or `python main.py`")

        st.markdown("**Models**")
        models = load_trained_models()
        if models:
            for m in models:
                st.success(f"✅ {m}")
        else:
            st.warning("No trained models found.\nRun `python main.py` first.")

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading dataset…"):
        df, load_error = load_and_prepare_data()

    if load_error:
        st.error(f"**Dataset error:** {load_error}")
        st.info("Run `python generate_dataset.py` then `python main.py` to set up the project.")
        st.stop()

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE: Overview
    # ═════════════════════════════════════════════════════════════════════════
    if page == "📊 Overview":
        st.markdown('<div class="section-header">Dataset Summary</div>',
                    unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{df['battery_id'].nunique()}</div>
                <div class="label">Batteries</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{len(df):,}</div>
                <div class="label">Total Cycles</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{df['soh'].min():.1f}%</div>
                <div class="label">Min SOH</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{df['soh'].max():.1f}%</div>
                <div class="label">Max SOH</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # SOH per battery
        st.markdown('<div class="section-header">SOH Degradation per Battery</div>',
                    unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(13, 5))
        battery_ids = df["battery_id"].unique()
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(battery_ids)))
        for bid, color in zip(battery_ids, colors):
            sub = df[df["battery_id"] == bid].sort_values("cycle")
            ax.plot(sub["cycle"], sub["soh"], label=f"Battery {bid}",
                    color=color, linewidth=2.0, alpha=0.9)
        ax.axhline(80, color="red", linestyle="--", linewidth=1.5,
                   label="EOL threshold (80%)")
        ax.set_xlabel("Cycle", fontsize=12)
        ax.set_ylabel("SOH (%)", fontsize=12)
        ax.set_title("State of Health Degradation", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, ncol=3)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown('<div class="section-header">Data Preview</div>',
                    unsafe_allow_html=True)
        display_cols = [c for c in ["battery_id", "cycle", "voltage", "temperature",
                                     "capacity", "soh", "rul"] if c in df.columns]
        st.dataframe(
            df[display_cols].head(50),
            use_container_width=True,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE: EDA Plots
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "🔍 EDA Plots":
        st.markdown('<div class="section-header">Exploratory Data Analysis</div>',
                    unsafe_allow_html=True)

        eda_figures = {
            "SOH vs Cycle": "soh_vs_cycle.png",
            "Capacity Degradation": "capacity_degradation.png",
            "Voltage over Cycles": "voltage_over_cycles.png",
            "Temperature Distribution": "temperature_dist.png",
            "Correlation Heatmap": "correlation_heatmap.png",
            "SOH Distribution": "soh_distribution.png",
            "Feature Boxplots": "feature_boxplots.png",
        }

        existing = {k: v for k, v in eda_figures.items()
                    if (FIGURES_DIR / v).exists()}

        if not existing:
            st.warning("EDA figures not found. Run `python main.py` to generate them.")
        else:
            for title, fname in existing.items():
                st.markdown(f"**{title}**")
                st.image(str(FIGURES_DIR / fname), use_container_width=True)
                st.markdown("---")

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE: Model Results
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "🤖 Model Results":
        st.markdown('<div class="section-header">Model Evaluation Results</div>',
                    unsafe_allow_html=True)

        metrics = load_metrics()
        if not metrics:
            st.warning("No metrics found. Run `python main.py` to train and evaluate models.")
        else:
            # Summary table
            rows = []
            for name, m in metrics.items():
                rows.append({
                    "Model": name.replace("_", " ").title(),
                    "MAE": f"{m['mae']:.4f}",
                    "RMSE": f"{m['rmse']:.4f}",
                    "R²": f"{m['r2']:.4f}",
                    "N Test": m["n_test"],
                })
            table_df = pd.DataFrame(rows)
            st.dataframe(table_df, use_container_width=True, hide_index=True)

            # Best model callout
            best = min(metrics.items(), key=lambda x: x[1]["mae"])
            st.success(f"🏆 Best model: **{best[0].replace('_', ' ').title()}** "
                       f"(MAE = {best[1]['mae']:.4f})")

            # Comparison chart
            if (FIGURES_DIR / "model_comparison.png").exists():
                st.markdown('<div class="section-header">Model Comparison Chart</div>',
                            unsafe_allow_html=True)
                st.image(str(FIGURES_DIR / "model_comparison.png"),
                         use_container_width=True)

            # Prediction plots
            st.markdown('<div class="section-header">Prediction Plots</div>',
                        unsafe_allow_html=True)
            for name in metrics:
                fig_path = FIGURES_DIR / f"predictions_{name}.png"
                if fig_path.exists():
                    st.markdown(f"**{name.replace('_', ' ').title()}**")
                    st.image(str(fig_path), use_container_width=True)
                    st.markdown("---")

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE: SOH Predictor
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "🎯 SOH Predictor":
        st.markdown('<div class="section-header">Real-Time SOH Predictor</div>',
                    unsafe_allow_html=True)

        if not models:
            st.warning("No trained models available. Run `python main.py` first.")
            st.stop()

        st.markdown("Enter battery measurements to predict the current State of Health:")

        col1, col2 = st.columns(2)
        with col1:
            battery_options = sorted(df["battery_id"].unique().tolist())
            selected_battery = st.selectbox("Battery ID", battery_options)
            cycle_val = st.number_input(
                "Cycle Number", min_value=1, max_value=1000, value=100)
            voltage_val = st.number_input(
                "Voltage (V)", min_value=2.0, max_value=4.5,
                value=3.75, step=0.01, format="%.3f")
            temperature_val = st.number_input(
                "Temperature (°C)", min_value=0.0, max_value=80.0,
                value=25.0, step=0.1)
            model_choice = st.selectbox("Model", list(models.keys()))

        with col2:
            if st.button("🔮 Predict SOH", type="primary", use_container_width=True):
                # Build feature row
                batt_ids = df["battery_id"].unique()
                batt_enc = {bid: i for i, bid in enumerate(sorted(batt_ids))}
                bid_enc  = batt_enc.get(selected_battery, 0)

                # Max cycle for this battery (for cycle_norm)
                max_cyc  = df[df["battery_id"] == selected_battery]["cycle"].max()
                min_cyc  = df[df["battery_id"] == selected_battery]["cycle"].min()
                cycle_norm = (cycle_val - min_cyc) / (max_cyc - min_cyc + 1e-9)

                # Use historical rolling averages for this battery up to cycle_val
                hist = df[
                    (df["battery_id"] == selected_battery) &
                    (df["cycle"] <= cycle_val)
                ].tail(5)

                v_roll_mean = hist["voltage_roll_mean"].mean() if len(hist) > 0 else voltage_val
                v_roll_std  = hist["voltage_roll_std"].mean()  if len(hist) > 0 else 0.0
                t_roll_mean = hist["temp_roll_mean"].mean()    if len(hist) > 0 else temperature_val
                v_diff      = 0.0
                t_diff      = 0.0
                if len(hist) >= 2:
                    v_diff = float(hist["voltage"].iloc[-1] - hist["voltage"].iloc[-2])
                    t_diff = float(hist["temperature"].iloc[-1] - hist["temperature"].iloc[-2])

                X_input = pd.DataFrame([{
                    "cycle":              cycle_val,
                    "voltage":            voltage_val,
                    "temperature":        temperature_val,
                    "battery_id_encoded": bid_enc,
                    "cycle_norm":         cycle_norm,
                    "voltage_roll_mean":  v_roll_mean,
                    "voltage_roll_std":   v_roll_std,
                    "temp_roll_mean":     t_roll_mean,
                    "voltage_diff":       v_diff,
                    "temp_diff":          t_diff,
                }])

                model = models[model_choice]
                pred_soh = float(model.predict(X_input)[0])
                pred_soh = np.clip(pred_soh, 0, 100)

                css_class = soh_color_class(pred_soh)
                status    = soh_status(pred_soh)

                st.markdown(f"""
                <div class="metric-card" style="margin-top:1rem;">
                    <div class="value"><span class="{css_class}">{pred_soh:.2f}%</span></div>
                    <div class="label">Predicted SOH</div>
                    <div style="margin-top:0.5rem; font-size:1.1rem;">{status}</div>
                </div>""", unsafe_allow_html=True)

                # Show historical SOH for context
                st.markdown("")
                st.markdown("**Historical SOH for this battery:**")
                hist_all = df[df["battery_id"] == selected_battery].sort_values("cycle")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(hist_all["cycle"], hist_all["soh"],
                        color="steelblue", linewidth=2, label="Historical SOH")
                ax.axvline(cycle_val, color="orange", linestyle="--",
                           linewidth=2, label=f"Current cycle ({cycle_val})")
                ax.scatter([cycle_val], [pred_soh], color="red", zorder=5,
                           s=100, label=f"Predicted SOH ({pred_soh:.1f}%)")
                ax.axhline(80, color="red", linestyle=":", linewidth=1, label="EOL (80%)")
                ax.set_xlabel("Cycle", fontsize=11)
                ax.set_ylabel("SOH (%)", fontsize=11)
                ax.set_title(f"Battery {selected_battery} — SOH History", fontsize=13)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.25)
                st.pyplot(fig)
                plt.close(fig)

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE: Feature Importance
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "📈 Feature Importance":
        st.markdown('<div class="section-header">Feature Importance Analysis</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        ### Why these features?

        | Feature | Type | Justification |
        |---------|------|---------------|
        | `cycle` | Raw | Direct measure of battery age |
        | `voltage` | Raw | Discharge voltage — measurable, degrades with age |
        | `temperature` | Raw | Operating temperature — affects degradation rate |
        | `battery_id_encoded` | Category | Cell chemistry/type identity |
        | `cycle_norm` | Engineered | Normalised age [0,1] per battery |
        | `voltage_roll_mean` | Engineered | 5-cycle rolling mean of voltage |
        | `voltage_roll_std` | Engineered | Voltage stability indicator |
        | `temp_roll_mean` | Engineered | Thermal stability indicator |
        | `voltage_diff` | Engineered | Rate of voltage change per cycle |
        | `temp_diff` | Engineered | Rate of temperature change per cycle |

        > **Excluded features**: `capacity`, `initial_capacity` (used to compute SOH directly),
        > `soh` (the target), `rul` (derived from future SOH trajectory).
        """)

        st.markdown('<div class="section-header">Feature Importance Charts</div>',
                    unsafe_allow_html=True)

        for model_name in ["random_forest", "xgboost"]:
            fig_path = FIGURES_DIR / f"feature_importance_{model_name}.png"
            if fig_path.exists():
                st.markdown(f"**{model_name.replace('_', ' ').title()}**")
                st.image(str(fig_path), use_container_width=True)
                st.markdown("---")
            else:
                st.info(f"Feature importance plot for {model_name} not found. "
                        "Run `python main.py` to generate.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<center style='color:gray; font-size:0.85rem;'>"
        "EV Battery Health Prediction · NASA Battery Dataset · "
        "Random Forest · XGBoost · MLP Neural Network"
        "</center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
