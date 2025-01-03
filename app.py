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
    "physical_activity_frequency": "How often do you engage in physical activity? (1 = Rarely, 2 = Sometimes, 3 = Frequently)",
    "height_meters": "Enter your height in meters (e.g., 1.75)",
    "num_meals_per_day": "Enter the average number of meals you eat per day (e.g., 3)",
    "Age": "Enter your age in years (e.g., 25)"
}

# Collect discrete field inputs
discrete_data = []
for field_name, help_text in discrete_fields.items():
    value = st.selectbox(
        f"{field_name.replace('_', ' ').title()}",
        [0, 1],
        help=help_text
    )
    discrete_data.append(value)

# Collect continuous field inputs
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

# Combine the discrete and continuous data
input_data = discrete_data + continuous_data
input_data_np = np.array(input_data).reshape(1, -1)  # Reshape to match model input format

# Predict button
if st.button("Predict Obesity"):
    try:
        # Scale the continuous features (continuous_data only)
        continuous_data = np.array(continuous_data).reshape(1, -1)
        continuous_data_scaled = loaded_scaler.transform(continuous_data)

        # Convert discrete_data into a 2D array to be compatible with continuous_data_scaled
        discrete_data_2d = np.array(discrete_data).reshape(1, -1)

        # Combine the discrete data (now 2D) and scaled continuous data
        new_data_scaled = np.hstack((discrete_data_2d, continuous_data_scaled))

        # Predict obesity
        obesity = loaded_model.predict(new_data_scaled)
        result = "Obese" if obesity[0] == 1 else "Non-obese"
        st.success(f"Predicted Obesity: {result}")
        
    except ValueError as e:
        st.error(f"Error: {e}")
