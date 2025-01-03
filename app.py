import streamlit as st
import joblib
import numpy as np

model_path = "model/Random Forest_model.pkl"
scaler_path = "model/scaler.pkl"

loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)

# Streamlit App
st.title("Obesity Prediction")

st.write("Enter the input values for the fields:")

# Input fields for the user
fields = ['emotional_eating', 'water_intake', 'freq_high_calorie_food', 'num_meals_per_day', 'height_meters', 'Age', 'family_history_overweight']

input_data = []
for field in fields:
    value = st.number_input(f"{field}", min_value=0.0, step=1e-3, format="%.2f")
    input_data.append(value)

# Predict button
if st.button("Predict Obesity"):
    # Scale the input data
    new_data = np.array(input_data).reshape(1, -1)
    new_data_scaled = loaded_scaler.transform(new_data)

    # Predict obesity
    obesity = loaded_model.predict(new_data_scaled)
    st.success(f"Predicted Obesity: {obesity[0]}")
