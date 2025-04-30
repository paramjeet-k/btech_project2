import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# — Page config —
st.set_page_config(page_title="UCS Predictor", layout="wide")

st.title("📊 UCS Prediction for Composite Mixes")
st.markdown("Upload your mix dataset (Excel or CSV), then predict UCS by manual input or by sample ID.")

# — File uploader (CSV or XLSX) —
uploaded = st.sidebar.file_uploader(
    "Upload your data (CSV or Excel)", type=["csv", "xlsx"]
)
if not uploaded:
    st.info("Waiting for you to upload a file…")
    st.stop()

# — Read data —
try:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

# — Validate columns —
required = ['MIX S. NO','Quartz (%)','Barite (%)','Detergent (%)',
            'Gypsum (%)','Cement (%)','Water (%)','UCS(Mpa)']
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# — Fit exponential model —
X = df[['Quartz (%)','Barite (%)','Detergent (%)','Gypsum (%)','Cement (%)','Water (%)']] / 100.0
y = df['UCS(Mpa)'].astype(float)
ln_y = np.log(y)
model = LinearRegression().fit(X, ln_y)
intercept_ln = model.intercept_
coeffs = dict(zip(X.columns, model.coef_))
A = np.exp(intercept_ln)

# — Show fitted equation —
eq = "UCS = {:.3e} * exp({})".format(
    A,
    " + ".join(f"{coeffs[c]:.4f}·{c}/100" for c in X.columns)
)
st.subheader("🔍 Fitted Transcendental Model")
st.code(eq)

# — Prediction mode —
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
    st.subheader(f"📋 Sample {sid}: Actual vs Predicted")
    st.write(f"- Actual UCS:    **{row['UCS(Mpa)']:.4f} MPa**")
    st.write(f"- Predicted UCS: **{ucs_pred:.4f} MPa**")

# — Optional: show data —
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Dataset")
    st.dataframe(df)
