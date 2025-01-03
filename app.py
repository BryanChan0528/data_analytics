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

# Predict button
if st.button("Predict Obesity"):
    try:
        # Separate discrete and continuous features
        discrete_data = np.array(discrete_data).reshape(1, -1)  # Discrete features don't need scaling
        continuous_data = np.array(continuous_data).reshape(1, -1)  # Only scale continuous (float) features
        
        # Debugging: Check shapes
        st.write(f"Discrete data shape: {discrete_data.shape}")
        st.write(f"Continuous data shape: {continuous_data.shape}")
        st.write(f"Scaler expected number of features for continuous data: {loaded_scaler.n_features_in_}")

        # Scale only the continuous features
        continuous_data_scaled = loaded_scaler.transform(continuous_data)

        # Combine scaled continuous data with discrete data
        new_data_scaled = np.hstack((discrete_data, continuous_data_scaled))
        
        # Debugging: Check final shape
        st.write(f"Final input shape after combining discrete and scaled continuous data: {new_data_scaled.shape}")

        # Predict obesity
        obesity = loaded_model.predict(new_data_scaled)
        result = "Obese" if obesity[0] == 1 else "Non-obese"
        st.success(f"Predicted Obesity: {result}")
    except ValueError as e:
        st.error(f"Error: {e}")
