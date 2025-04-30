import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Page Config ---
st.set_page_config(page_title="UCS Predictor", layout="wide")

st.title("📊 UCS Prediction for Composite Mixes")
st.markdown("This app loads a predefined CSV (`Book3.csv`) and predicts UCS by manual input or by sample ID.")

# --- Load dataset from local CSV ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Book3.csv')
    except FileNotFoundError:
        st.error("`Book3.csv` not found. Please ensure the file is in the app directory.")
        st.stop()
    return df

df = load_data()

# --- Validate columns ---
required = ['MIX S. NO','Quartz (%)','Barite (%)','Detergent (%)',
            'Gypsum (%)','Cement (%)','Water (%)','UCS(Mpa)']
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns in Book3.csv: {missing}")
    st.stop()

# --- Fit exponential model ---
X = df[['Quartz (%)','Barite (%)','Detergent (%)','Gypsum (%)','Cement (%)','Water (%)']] / 100.0
y = df['UCS(Mpa)'].astype(float)
ln_y = np.log(y)
model = LinearRegression().fit(X, ln_y)
intercept_ln = model.intercept_
coeffs = dict(zip(X.columns, model.coef_))
A = np.exp(intercept_ln)

# --- Show fitted equation ---
eq = (
    f"UCS = {A:.3e} * exp(" +
    " + ".join(f"{coeffs[c]:.4f}·{c}/100" for c in X.columns) + ")"
)
st.subheader("🔍 Fitted Transcendental Model")
st.code(eq)

# --- Prediction mode ---
mode = st.sidebar.radio("Prediction Mode", ["Manual Input", "By Sample ID"])

if mode == "Manual Input":
    st.sidebar.subheader("Enter your mix percentages")
    vals = {c: st.sidebar.number_input(c, min_value=0.0, max_value=100.0,
                                        value=float(df[c].mean()), step=0.1)
            for c in X.columns}
    total = sum(vals.values())
    if total != 100:
        st.sidebar.warning(f"Total = {total:.1f}%, should sum to 100%")
    exp_val = sum(coeffs[c]*vals[c] for c in X.columns)
    ucs_pred = A * np.exp(exp_val)
    st.subheader("🏷 Predicted UCS")
    st.metric("UCS (MPa)", f"{ucs_pred:.4f}")

else:  # By Sample ID
    st.sidebar.subheader("Select a sample")
    sid = st.sidebar.selectbox("MIX S. NO", df['MIX S. NO'].unique())
    row = df[df['MIX S. NO'] == sid].iloc[0]
    exp_val = sum(coeffs[c]*row[c] for c in X.columns)
    ucs_pred = A * np.exp(exp_val)
    st.subheader(f"📋 Sample {sid}: Actual vs Predicted UCS")
    st.write(f"- Actual UCS:    **{row['UCS(Mpa)']:.4f} MPa**")
    st.write(f"- Predicted UCS: **{ucs_pred:.4f} MPa**")

# --- Optional: show raw data ---
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Dataset")
    st.dataframe(df)
