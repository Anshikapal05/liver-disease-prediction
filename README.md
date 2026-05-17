# Liver Disease Prediction & Mortality Risk Predictor

Streamlit app for **two-stage ICU risk prediction** on PhysioNet MIMIC-II biomarkers (n=1750, 19.1% mortality):

1. **Liver disease** — XGBoost (ROC-AUC 0.9998)  
2. **Mortality risk** — SVM (ROC-AUC 0.7389) → Low / Medium / High using data-driven thresholds (8.4% / 36.1%)

> **Disclaimer:** For research purposes only. Not for clinical use.

## Quick start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Verify models:

```bash
python predictor.py
```

Expected outputs for built-in sample patients:

| Patient | Liver | Mortality | Risk |
|---------|-------|-----------|------|
| A — Healthy | 0.32% | 6.72% | Low |
| B — Moderate mortality risk | 99.98% | 59.36% | High |
| C — Severe multi-organ failure | 99.99% | 44.07% | High |

Use **Load Sample Patient** in the sidebar to check these in the UI.

## Project layout

```
├── streamlit_app.py      # Main app (single patient, batch CSV, model info)
├── predictor.py          # Load models, predict, validate inputs
├── requirements.txt
├── sample_patients.csv   # Example batch file (3 rows)
└── saved_models/         # Pre-trained joblib artifacts (commit to Git)
    ├── liver_disease_model.pkl
    ├── mortality_model.pkl
    ├── scaler_liver.pkl
    ├── scaler_mortality.pkl
    ├── thresholds.pkl
    └── feature_names.pkl
```

**Features (raw, unscaled):** Bilirubin, ALT, AST, ALP, Albumin, Creatinine, BUN, Na, K, HCO3, Platelets, HR, Age

## Batch prediction

Upload a CSV with the 13 columns above. Download adds: `Liver_Prob`, `Liver_Label`, `Mortality_Prob`, `Risk_Category`.

## Training notebook

Model development and evaluation: `Final_Liver.ipynb`
