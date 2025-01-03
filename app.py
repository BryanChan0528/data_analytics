import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
model_path = "model/Random Forest_model.pkl"
scaler_path = "model/scaler.pkl"

loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)

# Streamlit App
st.title("Obesity Prediction")

st.write("Enter the input values for the following parameters:")

# Input fields with user-friendly display names and placeholders
fields = {
    "Physical Activity Frequency": "How many times per week do you engage in physical activities? (e.g., 3)",
    "Height (Meters)": "Enter your height in meters (e.g., 1.75)",
    "Number of Meals per Day": "Enter the average number of meals you eat per day (e.g., 3)",
    "Family History of Overweight": "Does your family have a history of being overweight? (0 = No, 1 = Yes)",
    "Age (Years)": "Enter your age in years (e.g., 25)"
}

input_data = []
for display_name, placeholder in fields.items():
    if "Family History" in display_name:
        # Discrete field (0 or 1)
        value = st.selectbox(f"{display_name}", [0, 1], help=placeholder)
    else:
        # Continuous field with placeholder
        value = st.number_input(f"{display_name}", min_value=0.0, step=0.1, format="%.3f", placeholder=placeholder)
    input_data.append(value)

# Predict button
if st.button("Predict Obesity"):
    # Prepare the data for prediction
    new_data = np.array(input_data).reshape(1, -1)
    new_data_scaled = loaded_scaler.transform(new_data)

    # Predict obesity
    obesity = loaded_model.predict(new_data_scaled)
    result = "Obese" if obesity[0] == 1 else "Non-obese"
    st.success(f"Predicted Obesity: {result}")
