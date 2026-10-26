# EV Battery Health Prediction Using Advanced Machine Learning

> **Academic ML project** — Predicting Lithium-ion battery State of Health (SOH)  
> using Random Forest, XGBoost, and MLP Neural Network on the NASA Battery cycle-level dataset.

---

## 📁 Project Structure

```
Battery health/
├── data/
│   └── battery_dataset.csv          # Battery cycle-level dataset (NASA-style)
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Loading, validation, cleaning
│   ├── eda.py                       # Exploratory Data Analysis (7 plots)
│   ├── feature_engineering.py       # Feature construction (no leakage)
│   ├── train.py                     # RF, XGBoost, MLP training
│   ├── evaluate.py                  # MAE, RMSE, R² + plots
│   └── utils.py                     # Shared helpers, paths
├── models/
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── mlp.joblib
├── outputs/
│   ├── figures/                     # All saved PNG plots
│   └── metrics.json                 # Evaluation results
├── main.py                          # End-to-end pipeline runner
├── app.py                           # Streamlit dashboard
├── generate_dataset.py              # Synthetic dataset generator (fallback)
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**Source**: `battery_cycle_level_dataset.csv` (NASA Battery PCoE dataset, cycle-level)

| Column | Type | Description |
|--------|------|-------------|
| `battery_id` | str | Battery cell identifier (B0005, B0006, …) |
| `cycle` | int | Discharge cycle number |
| `voltage` | float | Average discharge voltage (V) |
| `temperature` | float | Cell temperature during discharge (°C) |
| `capacity` | float | Discharge capacity (Ah) |
| `soh` | float | State of Health (original: ratio; loaded as %) |
| `rul` | int | Remaining Useful Life (cycles) |

- **34 batteries**, up to 168 cycles each, **1,415 total rows**
- No missing values in the raw file

---

## 🚫 Data Leakage Policy

This project strictly avoids target leakage. The following columns are **excluded** from model inputs:

| Column | Reason excluded |
|--------|----------------|
| `soh` | This is the prediction target |
| `capacity` | SOH = capacity / initial_capacity → trivial reconstruction |
| `initial_capacity` | Denominator of the SOH formula; encodes per-battery identity |
| `rul` | Derived from future SOH trajectory; unavailable at inference time |

---

## 🔬 Features Used

| Feature | Source |
|---------|--------|
| `cycle` | Raw |
| `voltage` | Raw |
| `temperature` | Raw |
| `battery_id_encoded` | Label-encoded from `battery_id` |
| `cycle_norm` | Engineered — cycle normalised [0, 1] per battery |
| `voltage_roll_mean` | Engineered — 5-cycle rolling mean of voltage |
| `voltage_roll_std` | Engineered — 5-cycle rolling std of voltage |
| `temp_roll_mean` | Engineered — 5-cycle rolling mean of temperature |
| `voltage_diff` | Engineered — cycle-to-cycle voltage difference |
| `temp_diff` | Engineered — cycle-to-cycle temperature difference |

---

## 🤖 Models

| Model | Library | Notes |
|-------|---------|-------|
| Random Forest | scikit-learn | 200 trees, min_samples_leaf=2 |
| XGBoost | xgboost | 300 rounds, lr=0.05, max_depth=6 |
| MLP Neural Network | scikit-learn | (128→64→32), ReLU, Adam, early stopping |

**Train/test split**: Battery-level split (≈80% batteries for training, ≈20% held out).  
No test battery is ever seen during training.

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify dataset

The dataset should be at `data/battery_dataset.csv`.  
If missing, copy from the source location:

```bash
copy "C:\Users\NABEEL\Downloads\battery_cycle_level_dataset.csv" "data\battery_dataset.csv"
```

Or generate a synthetic version:

```bash
python generate_dataset.py
```

---

## 🚀 Running the Pipeline

```bash
python main.py
```

This will:
1. Load and inspect the dataset
2. Run EDA (saves 7 figures to `outputs/figures/`)
3. Engineer features (with leakage prevention)
4. Split by `battery_id` (train/test)
5. Train Random Forest, XGBoost, and MLP
6. Evaluate all models (MAE, RMSE, R²)
7. Save models to `models/` and metrics to `outputs/metrics.json`
8. Generate prediction and comparison plots

---

## 🖥️ Streamlit Dashboard

```bash
streamlit run app.py
```

Dashboard pages:
- **📊 Overview** — Dataset stats, SOH degradation chart, data preview
- **🔍 EDA Plots** — All 7 exploratory analysis figures
- **🤖 Model Results** — Evaluation table, comparison chart, prediction plots
- **🎯 SOH Predictor** — Enter live sensor readings → get SOH prediction
- **📈 Feature Importance** — Ranked feature importance for RF & XGBoost

---

## 📈 Historical Benchmark (to verify)

The following results were previously reported and should be verified by rerunning `main.py`:

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Random Forest | 7.4248 | 12.814 | −0.0963 |

> **Note**: These poor R² / RMSE results for Random Forest may be explained by the battery-level split — test batteries may have different degradation trajectories from training batteries, making cross-battery generalisation hard with only cycle/voltage/temperature as inputs. XGBoost and MLP results should be compared on the same test set.

---

## 📦 Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.25.0
joblib>=1.3.0
```

---

## 🗂️ Output Files

| File | Description |
|------|-------------|
| `outputs/metrics.json` | MAE, RMSE, R² for all models |
| `outputs/figures/soh_vs_cycle.png` | SOH degradation per battery |
| `outputs/figures/capacity_degradation.png` | Capacity fade |
| `outputs/figures/correlation_heatmap.png` | Feature correlation matrix |
| `outputs/figures/model_comparison.png` | Side-by-side metric comparison |
| `outputs/figures/predictions_*.png` | Predicted vs actual, residuals |
| `outputs/figures/feature_importance_*.png` | Feature importance charts |
| `models/*.joblib` | Saved trained models |
