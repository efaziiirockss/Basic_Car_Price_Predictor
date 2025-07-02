import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open("LrModel.pkl", "rb") as f:
    pipe = pickle.load(f)


st.title("🚗 Car Price Predictor")

# User inputs
name = st.text_input("Enter Model Name")
company = st.selectbox("Select Car Company", ['Hyundai', 'Mahindra', 'Ford', 'Maruti', 'Skoda', 'Audi', 'Toyota',
       'Renault', 'Honda', 'Datsun', 'Mitsubishi', 'Tata', 'Volkswagen',
       'Chevrolet', 'Mini', 'BMW', 'Nissan', 'Hindustan', 'Fiat', 'Force',
       'Mercedes', 'Land', 'Jaguar', 'Jeep', 'Volvo'])
year = st.number_input("Year of Purchase", min_value=1995, max_value=2019, step=1)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "LPG"])
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=100)

if st.button("Predict Price"):
    input_df = pd.DataFrame({
        "company": [company],
        "name": [name],
        "year": [year],
        "fuel_type": [fuel_type],
        "kms_driven": [kms_driven]
    })

    prediction = pipe.predict(input_df)[0]
    st.success(f"Estimated Car Price: ₹ {prediction:,.2f}")
