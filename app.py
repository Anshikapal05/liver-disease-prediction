import streamlit as st
import pickle
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Liver Disease Predictor",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------
# Load Models
# ----------------------------
rf = pickle.load(open('disease_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

rf_m = pickle.load(open('mortality_model.pkl', 'rb'))
scaler_m = pickle.load(open('scaler_m.pkl', 'rb'))

disease_features = pickle.load(open('disease_features.pkl', 'rb'))
mortality_features = pickle.load(open('mortality_features.pkl', 'rb'))

# ----------------------------
# Custom Styling
# ----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Title Section
# ----------------------------
st.title("🩺 Liver Disease & Mortality Prediction System")
st.markdown("### AI-powered healthcare prediction system")

st.write("---")

# ----------------------------
# Layout (2 Columns)
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tb = st.number_input("Total Bilirubin", 0.0)
    db = st.number_input("Direct Bilirubin", 0.0)
    alk = st.number_input("Alkaline Phosphotase", 0.0)

with col2:
    sgpt = st.number_input("SGPT", 0.0)
    sgot = st.number_input("SGOT", 0.0)
    tp = st.number_input("Total Proteins", 0.0)
    alb = st.number_input("Albumin", 0.0)
    agr = st.number_input("A/G Ratio", 0.0)

# Convert gender
gender_val = 1 if gender == "Male" else 0

# ----------------------------
# Prediction Button
# ----------------------------
def create_pdf(age, gender, disease, final_risk):

    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    
    result = "No Disease"
    risk = "N/A"
    
    if disease == 1:
        result = "Disease Detected"
        risk = "High Risk" if final_risk == 1 else "Low Risk"    
    content = [
        Paragraph(f"<b>Age:</b> {age}", styles["Normal"]),
        Paragraph(f"<b>Gender:</b> {gender}", styles["Normal"]),
        Paragraph(f"<b>Result:</b> {result}", styles["Normal"]),
        Paragraph(f"<b>Mortality Risk:</b> {risk}", styles["Normal"]),
    ]
    
    doc.build(content)
    
    with open("report.pdf", "rb") as f:
        return f.read()


if st.button("🔍 Predict"):

    data = [age, gender_val, tb, db, alk, sgpt, sgot, tp, alb, agr]
    
    input_df = pd.DataFrame([data], columns=disease_features)

    # Disease Prediction
    input_scaled = scaler.transform(input_df)
    disease = rf.predict(input_scaled)[0]

    st.write("---")

    if disease == 1:
        st.error("⚠️ Liver Disease Detected")

        # ----------------------------
        # HYBRID RISK LOGIC
        # ----------------------------
        risk_score = 0

        # Strict rule control
        if risk_score >= 2:
            final_risk = 1

        elif risk_score == 1:
            # medium → let ML decide
            input_mort = input_df[mortality_features]
            input_scaled_m = scaler_m.transform(input_mort)
            final_risk = rf_m.predict(input_scaled_m)[0]

        else:
            # clearly low case
            final_risk = 0

        # ----------------------------
        # OUTPUT
        # ----------------------------
        if final_risk == 1:
            st.error("🔴 High Mortality Risk")
        else:
            st.warning("🟡 Low Mortality Risk")

    else:
        st.success("✅ No Liver Disease Detected")
        final_risk = None

    # ----------------------------
    # TEXT REPORT
    # ----------------------------
    report = f"""
LIVER DISEASE REPORT

Age: {age}
Gender: {gender}

Result:
{'Disease Detected' if disease == 1 else 'No Disease'}

Mortality Risk:
{'High' if disease==1 and final_risk==1 else 'Low' if disease==1 else 'N/A'}
"""

    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name="liver_report.txt",
        mime="text/plain"
    )

    # ----------------------------
    # PDF REPORT
    # ----------------------------
    pdf = create_pdf(age, gender, disease, final_risk)

    st.download_button(
        label="📥 Download PDF Report",
        data=pdf,
        file_name="report.pdf",
        mime="application/pdf"
    )

    # ----------------------------
    # GRAPH
    # ----------------------------
    col_graph1, col_graph2, col_graph3 = st.columns([1,2,1])

    with col_graph2:
        st.write("### 📊 Patient Data Visualization")
        
        fig, ax = plt.subplots(figsize=(5,3))

        features = ['TB', 'DB', 'ALK', 'SGPT', 'SGOT']
        values = [tb, db, alk, sgpt, sgot]

        ax.bar(features, values)

        ax.set_title("Key Liver Parameters", fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

        st.pyplot(fig)

# ----------------------------
# Footer
# ----------------------------
st.write("---")
st.markdown("Developed using Machine Learning & Streamlit")
