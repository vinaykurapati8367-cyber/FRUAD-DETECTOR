import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.set_page_config(page_title="AI Insurance Fraud Detection", page_icon="🔍")

st.title("🔍 AI Insurance Fraud Detection System")
st.write("Detect fraudulent insurance claims using Machine Learning")

data = {
    "claim_amount":[5000,20000,15000,3000,25000],
    "previous_claims":[0,2,1,0,3],
    "fraud":[0,1,1,0,1]
}

df = pd.DataFrame(data)

X = df[["claim_amount","previous_claims"]]
y = df["fraud"]

model = RandomForestClassifier()
model.fit(X,y)

st.subheader("Enter Claim Details")

claim_amount = st.number_input("Claim Amount", min_value=0)
previous_claims = st.number_input("Previous Claims", min_value=0)

if st.button("Check Fraud"):

    prediction = model.predict([[claim_amount,previous_claims]])
    probability = model.predict_proba([[claim_amount,previous_claims]])

    fraud_prob = probability[0][1]*100

    st.write(f"Fraud Probability: {fraud_prob:.2f}%")

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Claim Detected")
    else:
        st.success("✅ Genuine Claim")

st.subheader("Sample Data Used For Training")

st.dataframe(df)

st.subheader("Claim Amount Distribution")

st.bar_chart(df["claim_amount"])