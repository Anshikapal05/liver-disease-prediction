# import streamlit as st
# import pickle
# import pandas as pd
# st.image("https://cdn-icons-png.flaticon.com/512/2966/2966485.png", width=100)
# # Load models
# rf = pickle.load(open('disease_model.pkl', 'rb'))
# scaler = pickle.load(open('scaler.pkl', 'rb'))

# rf_m = pickle.load(open('mortality_model.pkl', 'rb'))
# scaler_m = pickle.load(open('scaler_m.pkl', 'rb'))

# disease_features = pickle.load(open('disease_features.pkl', 'rb'))
# mortality_features = pickle.load(open('mortality_features.pkl', 'rb'))

# # Title
# st.title("Liver Disease & Mortality Prediction System")

# st.write("Enter patient details:")

# # Input fields
# age = st.number_input("Age")
# gender = st.selectbox("Gender (Male=1, Female=0)", [1,0])
# tb = st.number_input("Total Bilirubin")
# db = st.number_input("Direct Bilirubin")
# alk = st.number_input("Alkaline Phosphotase")
# sgpt = st.number_input("SGPT")
# sgot = st.number_input("SGOT")
# tp = st.number_input("Total Proteins")
# alb = st.number_input("Albumin")
# agr = st.number_input("Albumin/Globulin Ratio")

# # Prediction button
# if st.button("Predict"):

#     data = [age, gender, tb, db, alk, sgpt, sgot, tp, alb, agr]
    
#     input_df = pd.DataFrame([data], columns=disease_features)
    
#     # Disease prediction
#     input_scaled = scaler.transform(input_df)
#     disease = rf.predict(input_scaled)[0]
    
#     if disease == 1:
#         input_mort = input_df[mortality_features]
#         input_scaled_m = scaler_m.transform(input_mort)
#         mortality = rf_m.predict(input_scaled_m)[0]
        
#         if mortality == 1:
#             st.error("Disease Detected - High Mortality Risk")
#         else:
#             st.warning("Disease Detected - Low Mortality Risk")
#     else:
#         st.success("No Liver Disease Detected")

import streamlit as st
import pickle
import pandas as pd

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
if st.button("🔍 Predict"):

    data = [age, gender_val, tb, db, alk, sgpt, sgot, tp, alb, agr]
    
    input_df = pd.DataFrame([data], columns=disease_features)

    # Disease Prediction
    input_scaled = scaler.transform(input_df)
    disease = rf.predict(input_scaled)[0]

    st.write("---")

    if disease == 1:
        st.error("⚠️ Liver Disease Detected")

        # Mortality Prediction
        input_mort = input_df[mortality_features]
        input_scaled_m = scaler_m.transform(input_mort)
        mortality = rf_m.predict(input_scaled_m)[0]

        if mortality == 1:
            st.error("🔴 High Mortality Risk")
        else:
            st.warning("🟡 Low Mortality Risk")

    else:
        st.success("✅ No Liver Disease Detected")

# ----------------------------
# Footer
# ----------------------------
st.write("---")
st.markdown("Developed using Machine Learning & Streamlit")