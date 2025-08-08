import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from tensorflow.keras.models import load_model


model = load_model("model.keras")
with open("preprocessing_pipeline_object.pkl", "rb") as file:
    ppln_prpc = pickle.load(file)

catg_geography = ["Germany", "Spain", "France"]
catg_gender = ["Male", "Female"]
catg_binary = [0, 1]

# Streamlit app
st.title("Customer Churn Prediction")

data = {
    "Geography": [st.selectbox("Geography", catg_geography)],
    "Gender": [st.selectbox("Gender", catg_gender)],
    "Age": [st.slider("Age", 18, 90)],
    "Balance": [st.number_input("Balance")],
    "CreditScore": [st.number_input("Credit Score")],
    "EstimatedSalary": [st.number_input("Salary")],
    "Tenure": [st.slider("Tenure", 0, 10)],
    "NumOfProducts": [st.slider("Number of Products", 1, 4)],
    "HasCrCard": [st.selectbox("Has Credit Card", catg_binary)],
    "IsActiveMember": [st.selectbox("Is Active Member", catg_binary)],
}


df_cust = pd.DataFrame(data)
df_cust_tf = ppln_prpc.transform(df_cust)

y_pred = model.predict(df_cust_tf)
y_pred = pd.Series(y_pred.flatten())
y_pred_prob = y_pred[0]
y_pred = (y_pred > 0.7).astype(int)[0]

if y_pred == 1:
    st.write(
        f"The customer is likely to leave. The predicted probability to leave is {y_pred_prob * 100:.2f} %."
    )
else:
    st.write(
        f"The customer is likely to stay. The predicted probability to leave is {y_pred_prob * 100:.2f} %."
    )
