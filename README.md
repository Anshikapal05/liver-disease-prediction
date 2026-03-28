# 🩺 Liver Disease & Mortality Prediction System

An AI-powered healthcare web application that predicts:
- ✅ Presence of Liver Disease  
- ⚠️ Mortality Risk (Low / High)  

Built using **Machine Learning + Streamlit**, this system helps in early detection and risk assessment of liver-related conditions.

---

## 🚀 Features

✔ Liver Disease Prediction using ML models  
✔ Mortality Risk Prediction (Hybrid: Rule-based + ML)  
✔ Interactive Web UI (Streamlit)  
✔ Patient Data Visualization 📊  
✔ Download Report (TXT + PDF)  
✔ Clean & User-Friendly Interface  

---

## 🧠 Technologies Used

- Python 🐍  
- Scikit-learn  
- Pandas & NumPy  
- Matplotlib & Seaborn  
- Streamlit (for UI)  
- ReportLab (PDF generation)  

---

## 📊 Machine Learning Models

### 🔹 Disease Prediction
- Decision Tree  
- Random Forest  

### 🔹 Mortality Prediction
- Hybrid Approach:
  - Rule-based logic (clinical thresholds)
  - Random Forest (ML fallback)

---

## ⚙️ How It Works

1. User enters patient details:
   - Age, Gender  
   - Bilirubin levels  
   - SGPT, SGOT  
   - Albumin, Proteins  

2. System predicts:
   - Liver Disease (Yes/No)

3. If disease detected:
   - Mortality Risk is calculated using:
     - Rule-based scoring  
     - ML model (if required)

4. Outputs:
   - Risk Level (Low / High)
   - Visual Graph
   - Downloadable Report

---

## 📥 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/liver-disease-prediction.git
cd liver-disease-prediction

# 🩺 Liver Disease & Mortality Prediction System

An AI-powered healthcare web application that predicts:
- ✅ Presence of Liver Disease  
- ⚠️ Mortality Risk (Low / High)  

Built using **Machine Learning + Streamlit**, this system helps in early detection and risk assessment of liver-related conditions.

---

## 🚀 Features

✔ Liver Disease Prediction using ML models  
✔ Mortality Risk Prediction (Hybrid: Rule-based + ML)  
✔ Interactive Web UI (Streamlit)  
✔ Patient Data Visualization 📊  
✔ Download Report (TXT + PDF)  
✔ Clean & User-Friendly Interface  

---

## 🧠 Technologies Used

- Python 🐍  
- Scikit-learn  
- Pandas & NumPy  
- Matplotlib & Seaborn  
- Streamlit (for UI)  
- ReportLab (PDF generation)  

---

## 📊 Machine Learning Models

### 🔹 Disease Prediction
- Decision Tree  
- Random Forest  

### 🔹 Mortality Prediction
- Hybrid Approach:
  - Rule-based logic (clinical thresholds)
  - Random Forest (ML fallback)

---

## ⚙️ How It Works

1. User enters patient details:
   - Age, Gender  
   - Bilirubin levels  
   - SGPT, SGOT  
   - Albumin, Proteins  

2. System predicts:
   - Liver Disease (Yes/No)

3. If disease detected:
   - Mortality Risk is calculated using:
     - Rule-based scoring  
     - ML model (if required)

4. Outputs:
   - Risk Level (Low / High)
   - Visual Graph
   - Downloadable Report

---

## 📥 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/liver-disease-prediction.git
cd liver-disease-prediction
Install dependencies:

pip install -r requirements.txt
▶️ Run the App
streamlit run app.py

Then open in browser:

http://localhost:8501
📁 Project Structure
├── app.py
├── disease_model.pkl
├── mortality_model.pkl
├── scaler.pkl
├── scaler_m.pkl
├── disease_features.pkl
├── mortality_features.pkl
├── requirements.txt
└── README.md
📊 Sample Input
🟢 Healthy Case
Age: 30
Bilirubin: 0.8
SGPT: 25
SGOT: 30
Albumin: 4.0
🔴 High Risk Case
Age: 65
Bilirubin: 4.5
SGOT: 200
Albumin: 2.5
🧪 Model Logic
High bilirubin + low albumin + high SGOT → High Risk
Otherwise → ML model decides


⭐ Future Improvements
Deep Learning Models (ANN)
Real-time hospital integration
Larger medical datasets
Risk percentage visualization