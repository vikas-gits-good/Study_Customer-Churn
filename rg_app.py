import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from tensorflow.keras.models import load_model


model = load_model("rg_model.keras")
with open("rg_ppln.pkl", "rb") as file:
    ppln_prpc = pickle.load(file)

catg_geography = ["Germany", "Spain", "France"]
catg_gender = ["Male", "Female"]
catg_binary = [0, 1]

# Streamlit app
st.title("Customer Salary Prediction")

data = {
    "Geography": [st.selectbox("Geography", catg_geography)],
    "Gender": [st.radio("Gender", catg_gender)],
    "Age": [st.slider("Age", 18, 90)],
    "Balance": [st.number_input("Balance")],
    "CreditScore": [st.number_input("Credit Score")],
    "Tenure": [st.slider("Tenure", 0, 10)],
    "NumOfProducts": [st.slider("Number of Products", 1, 4)],
    "HasCrCard": [st.radio("Has Credit Card", catg_binary)],
    "IsActiveMember": [st.radio("Is Active Member", catg_binary)],
}


df_cust = pd.DataFrame(data)
df_cust_tf = ppln_prpc.transform(df_cust)

y_pred = model.predict(df_cust_tf)
y_pred = pd.Series(y_pred.flatten())
y_pred = y_pred[0]

st.write(f"The predicted salary is ${y_pred:,.2f}.")
