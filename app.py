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

# Input fields with user-friendly display names
discrete_fields = {
    "Family History of Overweight": "Does your family have a history of being overweight? (0 = No, 1 = Yes)",
    "Emotional Eating": "Do you eat due to emotional reasons? (0 = No, 1 = Yes)"
}

continuous_fields = {
    "Water Intake (Liters)": "Enter your daily water intake in liters (e.g., 1.5)",
    "Frequency of High-Calorie Food Consumption": "How often do you consume high-calorie foods? (1 = Rarely, 2 = Sometimes, 3 = Frequently)",
    "Number of Meals per Day": "Enter the average number of meals you eat per day (e.g., 3)",
    "Height (Meters)": "Enter your height in meters (e.g., 1.75)",
    "Age (Years)": "Enter your age in years (e.g., 25)"
}

# Collect discrete field inputs
discrete_data = []
for display_name, help_text in discrete_fields.items():
    value = st.selectbox(f"{display_name}", [0, 1], help=help_text)
    discrete_data.append(value)

# Collect continuous field inputs
continuous_data = []
for display_name, placeholder in continuous_fields.items():
    value = st.number_input(f"{display_name}", min_value=0.0, step=0.1, format="%.3f", placeholder=placeholder)
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
    st.success(f"Predicted Obesity: {'Yes' if obesity[0] == 1 else 'No'}")
