"""
Liver Disease & Mortality Risk Predictor — Streamlit application.
Run: streamlit run streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from predictor import (
    FEATURES,
    NORMAL_RANGES,
    SAMPLE_PATIENTS,
    get_risk_color,
    is_outside_normal,
    load_models,
    predict_batch,
    predict_patient,
    validate_inputs,
)

DISCLAIMER = (
    "**Disclaimer:** For research purposes only. "
    "Model trained on PhysioNet MIMIC-II ICU dataset (n=1750, 19.1% mortality)."
)

INPUT_SPECS: dict[str, dict] = {
    "Bilirubin": {"min": 0.0, "max": 30.0, "step": 0.1, "format": "%.1f"},
    "ALT": {"min": 0.0, "max": 500.0, "step": 1.0, "format": "%.0f"},
    "AST": {"min": 0.0, "max": 600.0, "step": 1.0, "format": "%.0f"},
    "ALP": {"min": 0.0, "max": 800.0, "step": 1.0, "format": "%.0f"},
    "Albumin": {"min": 0.0, "max": 6.0, "step": 0.1, "format": "%.1f"},
    "Creatinine": {"min": 0.0, "max": 15.0, "step": 0.1, "format": "%.1f"},
    "BUN": {"min": 0.0, "max": 150.0, "step": 1.0, "format": "%.0f"},
    "Na": {"min": 100.0, "max": 180.0, "step": 1.0, "format": "%.0f"},
    "K": {"min": 0.0, "max": 10.0, "step": 0.1, "format": "%.1f"},
    "HCO3": {"min": 5.0, "max": 45.0, "step": 1.0, "format": "%.0f"},
    "Platelets": {"min": 0.0, "max": 600.0, "step": 1.0, "format": "%.0f"},
    "HR": {"min": 20.0, "max": 220.0, "step": 1.0, "format": "%.0f"},
    "Age": {"min": 0.0, "max": 120.0, "step": 1.0, "format": "%.0f"},
}

FORM_SECTIONS = {
    "Liver Enzymes": ["Bilirubin", "ALT", "AST", "ALP", "Albumin"],
    "Renal Function": ["Creatinine", "BUN"],
    "Electrolytes": ["Na", "K", "HCO3"],
    "Clinical": ["Platelets", "HR", "Age"],
}

LIVER_MODEL_TABLE = [
    ["XGBoost", "0.9998", "0.9999", "0.9989±0.0012"],
    ["RandomForest", "0.9997", "0.9998", "0.9986±0.0012"],
    ["SVM", "0.9799", "0.9894", "0.9714±0.0078"],
    ["LogReg", "0.9730", "0.9851", "0.9572±0.0087"],
    ["NaiveBayes", "0.9612", "0.9794", "0.9604±0.0056"],
    ["KNN", "0.9271", "0.9618", "0.9490±0.0117"],
]

MORTALITY_MODEL_TABLE = [
    ["SVM", "0.7389", "0.3686", "0.8933±0.0138"],
    ["RandomForest", "0.7322", "0.3968", "0.9196±0.0083"],
    ["LogReg", "0.7144", "0.3936", "0.7738±0.0177"],
    ["XGBoost", "0.7092", "0.3722", "0.9428±0.0095"],
    ["NaiveBayes", "0.7080", "0.4197", "0.7332±0.0074"],
    ["KNN", "0.6855", "0.3233", "0.9383±0.0102"],
]


def _init_session_defaults() -> None:
    default = SAMPLE_PATIENTS["A — Healthy Young Adult"]
    if "inputs" not in st.session_state:
        st.session_state.inputs = dict(default)
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    for feature in FEATURES:
        key = f"input_{feature}"
        if key not in st.session_state:
            st.session_state[key] = float(st.session_state.inputs.get(feature, default[feature]))


def _help_text(feature: str) -> str:
    lo, hi, unit = NORMAL_RANGES[feature]
    return f"Normal: {lo}–{hi} {unit}"


def _render_disclaimer() -> None:
    st.markdown("---")
    st.caption(DISCLAIMER.replace("**Disclaimer:** ", ""))


def _number_input(feature: str, container) -> float:
    spec = INPUT_SPECS[feature]
    lo, hi, unit = NORMAL_RANGES[feature]
    value = container.number_input(
        label=f"{feature} ({unit})",
        min_value=spec["min"],
        max_value=spec["max"],
        step=spec["step"],
        format=spec["format"],
        help=_help_text(feature),
        key=f"input_{feature}",
    )
    st.session_state.inputs[feature] = value
    if is_outside_normal(feature, value):
        container.markdown(
            f'<p style="color:#e74c3c;font-size:0.85rem;margin-top:-0.5rem;">'
            f"Outside normal range ({lo}–{hi} {unit})</p>",
            unsafe_allow_html=True,
        )
    return value


def _render_input_form() -> dict[str, float]:
    patient: dict[str, float] = {}
    for section, features in FORM_SECTIONS.items():
        with st.expander(section, expanded=True):
            cols = st.columns(2 if len(features) > 2 else len(features))
            for i, feature in enumerate(features):
                with cols[i % len(cols)]:
                    patient[feature] = _number_input(feature, st)
    return patient


def _render_liver_result(liv_prob: float, liv_label: int) -> None:
    liv_pct = liv_prob * 100
    color = "#e74c3c" if liv_label else "#2ecc71"
    text = "Liver Disease Detected" if liv_label else "No Liver Disease"
    st.markdown(
        f'<div style="background:{color}22;border-left:6px solid {color};'
        f'padding:1rem 1.25rem;border-radius:8px;margin-bottom:1rem;">'
        f'<h3 style="color:{color};margin:0;">{text}</h3>'
        f'<p style="font-size:1.4rem;margin:0.5rem 0 0 0;">'
        f"Probability: <strong>{liv_pct:.2f}%</strong></p></div>",
        unsafe_allow_html=True,
    )
    st.progress(min(liv_prob, 1.0))
    st.caption(f"Liver disease probability: {liv_pct:.2f}%")


def _render_mortality_result(mort_prob: float, risk: str, low_t: float, high_t: float) -> None:
    mort_pct = mort_prob * 100
    color = get_risk_color(risk)
    st.markdown(
        f'<div style="background:{color}22;border-left:6px solid {color};'
        f'padding:1rem 1.25rem;border-radius:8px;margin-bottom:1rem;">'
        f'<h3 style="color:{color};margin:0;">{risk}</h3>'
        f'<p style="font-size:1.4rem;margin:0.5rem 0 0 0;">'
        f"Mortality probability: <strong>{mort_pct:.2f}%</strong></p></div>",
        unsafe_allow_html=True,
    )
    st.progress(min(mort_prob, 1.0))
    st.markdown(
        f"""
**Risk stratification (data-driven thresholds):**
- **Low Risk:** &lt; {low_t * 100:.1f}% predicted mortality
- **Medium Risk:** {low_t * 100:.1f}% – {high_t * 100:.1f}% predicted mortality
- **High Risk:** &gt; {high_t * 100:.1f}% predicted mortality

*Thresholds derived from 33rd/67th percentile of the model's predicted probability distribution.*
        """
    )


def _render_clinical_alerts(patient: dict[str, float]) -> None:
    warnings = validate_inputs(patient)
    if not warnings:
        st.success("All biomarkers within normal reference ranges.")
        return
    for w in warnings:
        color = "#e74c3c" if w["severity"] == "critical" else "#f39c12"
        icon = "🔴" if w["severity"] == "critical" else "🟠"
        st.markdown(
            f'<p style="color:{color};margin:0.25rem 0;">{icon} {w["message"]} '
            f'(<em>{w["severity"]}</em>)</p>',
            unsafe_allow_html=True,
        )


def page_single_patient(artifacts: dict) -> None:
    _init_session_defaults()
    low_t = artifacts["low_thresh"]
    high_t = artifacts["high_thresh"]

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("Patient Biomarkers")
        patient = _render_input_form()
        if st.button("Predict", type="primary", use_container_width=True):
            liv_prob, liv_label, mort_prob, risk = predict_patient(patient, artifacts)
            st.session_state.prediction = {
                "patient": patient,
                "liv_prob": liv_prob,
                "liv_label": liv_label,
                "mort_prob": mort_prob,
                "risk": risk,
            }

    with col_out:
        st.subheader("Prediction Results")
        if st.session_state.prediction is None:
            st.info("Enter biomarkers and click **Predict** to see results.")
        else:
            pred = st.session_state.prediction
            st.markdown("#### Stage 1 — Liver Disease")
            _render_liver_result(pred["liv_prob"], pred["liv_label"])
            st.markdown("#### Stage 2 — Mortality Risk")
            _render_mortality_result(
                pred["mort_prob"], pred["risk"], low_t, high_t
            )
            st.markdown("#### Clinical Alerts")
            _render_clinical_alerts(pred["patient"])
            st.markdown("#### Disclaimer")
            st.warning(
                "For research purposes only. Not for clinical use."
            )

    _render_disclaimer()


def page_batch(artifacts: dict) -> None:
    st.subheader("Batch Prediction")
    st.markdown(
        "Upload a CSV with exactly these 13 columns (raw unscaled values): "
        f"`{', '.join(FEATURES)}`"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file to run batch predictions.")
        _render_disclaimer()
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        _render_disclaimer()
        return

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        _render_disclaimer()
        return

    df = df[FEATURES].copy()
    with st.spinner("Running predictions…"):
        results = predict_batch(df, artifacts)

    st.success(f"Processed {len(results)} patients.")
    st.dataframe(results, use_container_width=True, height=400)

    st.markdown("#### Summary Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Liver Prob (%)", f"{results['Liver_Prob'].mean():.2f}")
    c2.metric("Mean Mortality Prob (%)", f"{results['Mortality_Prob'].mean():.2f}")
    c3.metric("Liver Disease (+)", int(results["Liver_Label"].sum()))
    risk_counts = results["Risk_Category"].value_counts()
    c4.metric("High Risk Count", int(risk_counts.get("High Risk", 0)))

    st.markdown("**Risk distribution:**")
    st.bar_chart(results["Risk_Category"].value_counts())

    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results CSV",
        data=csv_bytes,
        file_name="liver_mortality_predictions.csv",
        mime="text/csv",
        type="primary",
    )
    _render_disclaimer()


def page_model_info(artifacts: dict) -> None:
    st.subheader("Model Comparison")
    st.markdown("#### Liver Disease Models")
    st.table(
        pd.DataFrame(
            LIVER_MODEL_TABLE,
            columns=["Model", "ROC-AUC", "PR-AUC", "CV Score"],
        )
    )
    st.markdown("#### Mortality Models")
    st.table(
        pd.DataFrame(
            MORTALITY_MODEL_TABLE,
            columns=["Model", "ROC-AUC", "PR-AUC", "CV Score"],
        )
    )

    st.subheader("Feature Importance (SHAP)")
    st.markdown(
        """
**Liver Disease — top features:**  
HR = 3.50, ALP = 1.73, Age = 1.69, BUN = 1.55, K = 1.31

**Mortality — top features:**  
Bilirubin = 0.109, AST = 0.079, Na = 0.076, Albumin = 0.064, BUN = 0.046
        """
    )

    st.subheader("Risk Stratification")
    low_t = artifacts["low_thresh"]
    high_t = artifacts["high_thresh"]
    st.markdown(
        f"""
Mortality risk categories use fixed thresholds from the test-set distribution:

| Category | Predicted mortality probability |
|----------|--------------------------------|
| Low Risk | &lt; {low_t * 100:.2f}% ({low_t:.4f}) |
| Medium Risk | {low_t * 100:.2f}% – {high_t * 100:.2f}% |
| High Risk | ≥ {high_t * 100:.2f}% ({high_t:.4f}) |

**Deployed models:** Liver = XGBoost (ROC-AUC 0.9998), Mortality = SVM (ROC-AUC 0.7389)
        """
    )
    _render_disclaimer()


def main() -> None:
    st.set_page_config(
        page_title="Liver Disease & Mortality Risk Predictor",
        page_icon="🏥",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp { font-family: 'Segoe UI', system-ui, sans-serif; }
        div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("ICU Risk Predictor")
        st.markdown(
            "Two-stage pipeline for **liver disease detection** and "
            "**ICU mortality risk stratification** using MIMIC-II biomarkers."
        )
        st.markdown("---")
        st.markdown("**Load Sample Patient**")
        for label, data in SAMPLE_PATIENTS.items():
            if st.button(label, use_container_width=True, key=f"sample_{label}"):
                st.session_state.inputs = dict(data)
                st.session_state.prediction = None
                for f, v in data.items():
                    st.session_state[f"input_{f}"] = float(v)
                st.rerun()

        st.markdown("---")
        st.markdown(
            """
**About**
- Liver Model: XGBoost | ROC-AUC: 0.9998
- Mortality: SVM | ROC-AUC: 0.7389
- Dataset: PhysioNet MIMIC-II (n=1750)
- Mortality rate: 19.1%
            """
        )

    artifacts = load_models()

    page = st.sidebar.radio(
        "Navigation",
        ["Single Patient", "Batch Prediction", "Model Info"],
        label_visibility="collapsed",
    )

    st.title("Liver Disease & Mortality Risk Predictor")

    if page == "Single Patient":
        page_single_patient(artifacts)
    elif page == "Batch Prediction":
        page_batch(artifacts)
    else:
        page_model_info(artifacts)


if __name__ == "__main__":
    main()
