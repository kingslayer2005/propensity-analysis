import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Insurance Policy Renewal Prediction")

st.title("🚗 Insurance Policy Renewal Prediction")
st.write("Upload a **preprocessed** CSV with the same feature columns used in training (the ones in your `X`).")

# Load model + training columns
model = joblib.load("models/final_model.pkl")
train_cols = joblib.load("models/train_columns.pkl")

uploaded = st.file_uploader("Upload CSV (with the same engineered columns as training X)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)

    # Align columns to training columns: add missing as 0, drop extras, reorder
    for c in train_cols:
        if c not in df.columns:
            df[c] = 0
    X_new = df.reindex(columns=train_cols)

    # Predict
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)[:, 1]
        pred = (proba >= 0.5).astype(int)
        out = df.copy()
        out["renewal_prob"] = proba
        out["renewal_pred"] = pred
        st.success("✅ Predictions computed (with probabilities).")
    else:
        pred = model.predict(X_new)
        out = df.copy()
        out["renewal_pred"] = pred
        st.warning("Model does not support predict_proba; showing class predictions only.")

    st.dataframe(out.head(20))
    st.download_button("Download predictions.csv", out.to_csv(index=False), "predictions.csv")
else:
    st.info("Upload a CSV to get predictions.")
