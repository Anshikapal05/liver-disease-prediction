"""
ICU Liver Disease & Mortality Risk — prediction module.
Loads pre-trained models and runs the two-stage pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import streamlit as st

# Feature order (must match training)
FEATURES = [
    "Bilirubin",
    "ALT",
    "AST",
    "ALP",
    "Albumin",
    "Creatinine",
    "BUN",
    "Na",
    "K",
    "HCO3",
    "Platelets",
    "HR",
    "Age",
]

# Clinical reference ranges: (min, max, unit)
NORMAL_RANGES: dict[str, tuple[float, float, str]] = {
    "Bilirubin": (0.2, 1.2, "mg/dL"),
    "ALT": (7.0, 40.0, "U/L"),
    "AST": (10.0, 40.0, "U/L"),
    "ALP": (44.0, 120.0, "U/L"),
    "Albumin": (3.5, 5.0, "g/dL"),
    "Creatinine": (0.6, 1.2, "mg/dL"),
    "BUN": (7.0, 20.0, "mg/dL"),
    "Na": (135.0, 145.0, "mEq/L"),
    "K": (3.5, 5.0, "mEq/L"),
    "HCO3": (22.0, 29.0, "mEq/L"),
    "Platelets": (150.0, 400.0, "x10³/µL"),
    "HR": (60.0, 100.0, "bpm"),
    "Age": (18.0, 100.0, "years"),
}

LOW_THRESH = 0.0842
HIGH_THRESH = 0.3610

RISK_COLORS = {
    "Low Risk": "#2ecc71",
    "Medium Risk": "#f39c12",
    "High Risk": "#e74c3c",
}

SAMPLE_PATIENTS = {
    "A — Healthy Young Adult": {
        "Bilirubin": 0.8,
        "ALT": 22,
        "AST": 20,
        "ALP": 70,
        "Albumin": 4.2,
        "Creatinine": 0.9,
        "BUN": 14,
        "Na": 140,
        "K": 4.0,
        "HCO3": 25,
        "Platelets": 220,
        "HR": 72,
        "Age": 32,
    },
    "B — Moderate Mortality Risk": {
        "Bilirubin": 3.5,
        "ALT": 150,
        "AST": 130,
        "ALP": 280,
        "Albumin": 3.0,
        "Creatinine": 1.3,
        "BUN": 28,
        "Na": 136,
        "K": 3.8,
        "HCO3": 21,
        "Platelets": 130,
        "HR": 88,
        "Age": 58,
    },
    "C — High Mortality Risk": {
        "Bilirubin": 8.2,
        "ALT": 320,
        "AST": 410,
        "ALP": 520,
        "Albumin": 2.1,
        "Creatinine": 3.8,
        "BUN": 65,
        "Na": 128,
        "K": 5.8,
        "HCO3": 14,
        "Platelets": 55,
        "HR": 118,
        "Age": 74,
    },
}

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "saved_models"


def _load_artifacts(model_dir: Path | str | None = None) -> dict[str, Any]:
    """Load all joblib artifacts from disk (uncached)."""
    base = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    liver_model = joblib.load(base / "liver_disease_model.pkl")
    mortality_model = joblib.load(base / "mortality_model.pkl")
    scaler_liver = joblib.load(base / "scaler_liver.pkl")
    scaler_mortality = joblib.load(base / "scaler_mortality.pkl")
    thresholds = joblib.load(base / "thresholds.pkl")
    feature_names = joblib.load(base / "feature_names.pkl")

    return {
        "liver_model": liver_model,
        "mortality_model": mortality_model,
        "scaler_liver": scaler_liver,
        "scaler_mortality": scaler_mortality,
        "thresholds": thresholds,
        "feature_names": list(feature_names),
        "low_thresh": float(thresholds["low_thresh"]),
        "high_thresh": float(thresholds["high_thresh"]),
    }


@st.cache_resource
def load_models() -> dict[str, Any]:
    """Load and cache models (call once per Streamlit session)."""
    return _load_artifacts()


def get_risk_color(risk_category: str) -> str:
    """Return hex color for a risk category label."""
    return RISK_COLORS.get(risk_category, "#95a5a6")


def validate_inputs(patient_data: dict[str, float]) -> list[dict[str, str]]:
    """
    Return warnings for biomarkers outside normal reference ranges.
    Each item: {feature, message, severity} where severity is 'borderline' or 'critical'.
    """
    warnings: list[dict[str, str]] = []
    for feature in FEATURES:
        if feature not in patient_data:
            continue
        value = float(patient_data[feature])
        lo, hi, unit = NORMAL_RANGES[feature]
        width = hi - lo
        if lo <= value <= hi:
            continue
        if value < lo:
            margin = lo - value
            severity = "borderline" if margin <= 0.2 * width else "critical"
            warnings.append(
                {
                    "feature": feature,
                    "message": (
                        f"{feature} = {value} {unit} is below normal "
                        f"({lo}–{hi} {unit})"
                    ),
                    "severity": severity,
                }
            )
        else:
            margin = value - hi
            severity = "borderline" if margin <= 0.2 * width else "critical"
            warnings.append(
                {
                    "feature": feature,
                    "message": (
                        f"{feature} = {value} {unit} is above normal "
                        f"({lo}–{hi} {unit})"
                    ),
                    "severity": severity,
                }
            )
    return warnings


def is_outside_normal(feature: str, value: float) -> bool:
    """True if value is outside the reference range for feature."""
    lo, hi, _ = NORMAL_RANGES[feature]
    return value < lo or value > hi


def predict_patient(
    patient_data: dict[str, float],
    artifacts: dict[str, Any] | None = None,
) -> tuple[float, int, float, str]:
    """
    Two-stage prediction on raw (unscaled) clinical values.

    Returns:
        liv_prob   — liver disease probability (0–1)
        liv_label  — 0 = no disease, 1 = disease
        mort_prob  — mortality probability (0–1)
        risk       — 'Low Risk' | 'Medium Risk' | 'High Risk'
    """
    if artifacts is None:
        artifacts = load_models()

    features = artifacts["feature_names"]
    low_t = artifacts["low_thresh"]
    high_t = artifacts["high_thresh"]

    X_raw = np.array([[patient_data[f] for f in features]])
    X_liv = artifacts["scaler_liver"].transform(X_raw)
    X_mort = artifacts["scaler_mortality"].transform(X_raw)

    liver_model = artifacts["liver_model"]
    mortality_model = artifacts["mortality_model"]

    liv_prob = float(liver_model.predict_proba(X_liv)[0][1])
    liv_label = int(liver_model.predict(X_liv)[0])
    mort_prob = float(mortality_model.predict_proba(X_mort)[0][1])

    if mort_prob < low_t:
        risk = "Low Risk"
    elif mort_prob < high_t:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return liv_prob, liv_label, mort_prob, risk


def predict_batch(
    df,
    artifacts: dict[str, Any] | None = None,
):
    """Run predict_patient on each row; return DataFrame with result columns."""
    import pandas as pd

    if artifacts is None:
        artifacts = load_models()

    rows = []
    for _, row in df.iterrows():
        patient = {f: float(row[f]) for f in FEATURES}
        liv_prob, liv_label, mort_prob, risk = predict_patient(patient, artifacts)
        rows.append(
            {
                "Liver_Prob": round(liv_prob * 100, 2),
                "Liver_Label": liv_label,
                "Mortality_Prob": round(mort_prob * 100, 2),
                "Risk_Category": risk,
            }
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


if __name__ == "__main__":
    EXPECTED = {
        "A — Healthy Young Adult": (0.32, 6.72, "Low Risk"),
        "B — Moderate Mortality Risk": (99.98, 59.36, "High Risk"),
        "C — High Mortality Risk": (99.99, 44.07, "High Risk"),
    }

    artifacts = _load_artifacts()
    print("Loaded models from:", DEFAULT_MODEL_DIR)
    print(f"Features: {artifacts['feature_names']}")
    print(f"Thresholds: low={artifacts['low_thresh']:.4f}, high={artifacts['high_thresh']:.4f}\n")

    all_ok = True
    for name, patient in SAMPLE_PATIENTS.items():
        liv_prob, liv_label, mort_prob, risk = predict_patient(patient, artifacts)
        liv_pct = round(liv_prob * 100, 2)
        mort_pct = round(mort_prob * 100, 2)
        exp_liv, exp_mort, exp_risk = EXPECTED[name]
        ok = liv_pct == exp_liv and mort_pct == exp_mort and risk == exp_risk
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"[{status}] {name}")
        print(f"  Liver: {liv_pct}% (expected {exp_liv}%)  label={liv_label}")
        print(f"  Mortality: {mort_pct}% (expected {exp_mort}%)  risk={risk} (expected {exp_risk})")
        print()

    if all_ok:
        print("All sample patients verified.")
    else:
        print("VERIFICATION FAILED — check models and prediction logic.")
        raise SystemExit(1)
