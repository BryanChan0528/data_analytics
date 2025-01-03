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

# Fields based on the selected features
discrete_fields = {
    "family_history_overweight": "Does your family have a history of being overweight? (0 = No, 1 = Yes)"
}

continuous_fields = {
    "height_meters": "Enter your height in meters (e.g., 1.75)",
    "num_meals_per_day": "Enter the average number of meals you eat per day (e.g., 3)",
    "Age": "Enter your age in years (e.g., 25)",
    "physical_activity_frequency": "How often do you engage in physical activity? (e.g., 1 = Rarely, 2 = Sometimes, 3 = Frequently)"
}

# Collect discrete field inputs (family_history_overweight remains as a selectbox)
discrete_data = []
for field_name, help_text in discrete_fields.items():
    value = st.selectbox(
        f"{field_name.replace('_', ' ').title()}",
        [0, 1],
        help=help_text
    )
    discrete_data.append(value)

# Collect continuous field inputs, including physical_activity_frequency as continuous
continuous_data = []
for field_name, placeholder in continuous_fields.items():
    value = st.number_input(
        f"{field_name.replace('_', ' ').title()}",
        min_value=0.0,
        step=0.1,
        format="%.3f",
        help=placeholder
    )
    continuous_data.append(value)

# Predict button
if st.button("Predict Obesity"):
    try:
        # Separate discrete and continuous features
        discrete_data = np.array(discrete_data).reshape(1, -1)  # Discrete features
        continuous_data = np.array(continuous_data).reshape(1, -1)  # Continuous features
        
        # Scale continuous features
        continuous_data_scaled = loaded_scaler.transform(continuous_data)

        # Combine scaled continuous data with discrete data
        new_data_scaled = np.hstack((discrete_data, continuous_data_scaled))

        # Predict obesity
        obesity = loaded_model.predict(new_data_scaled)
        result = "Obese" if obesity[0] == 1 else "Non-obese"
        st.success(f"Predicted Obesity: {result}")
    except ValueError as e:
        st.error(f"Error: {e}")
