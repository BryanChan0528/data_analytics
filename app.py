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

st.write("Enter the input values for the fields:")

# Input fields for the user
# Discrete fields (0 or 1)
discrete_fields = {
    "family_history_overweight": "Select 0 for No, 1 for Yes",
    "emotional_eating": "Select 0 for No, 1 for Yes"
}

# Continuous fields with placeholders
continuous_fields = {
    "water_intake": "Enter water intake in liters (e.g., 1.5)",
    "freq_high_calorie_food": "Enter frequency (1-3; 1 = Rarely, 2 = Sometimes, 3 = Frequently)",
    "num_meals_per_day": "Enter number of meals per day (e.g., 3)",
    "height_meters": "Enter height in meters (e.g., 1.75)",
    "Age": "Enter age in years (e.g., 25)"
}

# Collect discrete field inputs
discrete_data = []
for field, help_text in discrete_fields.items():
    value = st.selectbox(f"{field} ({help_text})", [0, 1])
    discrete_data.append(value)

# Collect continuous field inputs
continuous_data = []
for field, placeholder in continuous_fields.items():
    value = st.number_input(f"{field}", min_value=0.0, step=0.1, format="%.3f", placeholder=placeholder)
    continuous_data.append(value)

# Combine inputs
input_data = discrete_data + continuous_data

# Predict button
if st.button("Predict Obesity"):
    # Transform the continuous data if needed
    # (Add any transformation logic here if required, e.g., log scaling or normalization)

    # Prepare the data for prediction
    new_data = np.array(input_data).reshape(1, -1)
    new_data_scaled = loaded_scaler.transform(new_data)

    # Predict obesity
    obesity = loaded_model.predict(new_data_scaled)
    st.success(f"Predicted Obesity: {obesity[0]}")
