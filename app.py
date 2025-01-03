import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model and scaler
model_path = "model/Random Forest_model.pkl"
scaler_path = "model/scaler.pkl"
loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)

# Streamlit App
st.title("Obesity Prediction")
st.write("Enter the input values for the following parameters:")

# Fields definitions
discrete_fields = {
    "family_history_overweight": "Does your family have a history of being overweight? (0 = No, 1 = Yes)"
}

continuous_fields = {
    "height_meters": {"text": "Enter your height in meters", "min": 1.0, "max": 2.5},
    "num_meals_per_day": {"text": "Enter average meals per day", "min": 1, "max": 10},
    "Age": {"text": "Enter your age in years", "min": 1, "max": 120},
    "physical_activity_frequency": {"text": "Physical activity frequency (1-5)", "min": 1, "max": 5}
}

# Collect inputs with validation
discrete_data = []
for field_name, help_text in discrete_fields.items():
    value = st.selectbox(f"{field_name.replace('_', ' ').title()}", [0, 1], help=help_text)
    discrete_data.append(value)

continuous_data = []
for field_name, field_info in continuous_fields.items():
    value = st.number_input(
        f"{field_name.replace('_', ' ').title()}",
        min_value=field_info['min'],
        max_value=field_info['max'],
        help=field_info['text']
    )
    continuous_data.append(value)

if st.button("Predict Obesity"):
    try:
        # Create DataFrame with proper column names for continuous fields
        continuous_df = pd.DataFrame([continuous_data], 
                                   columns=continuous_fields.keys())
        
        # Scale continuous features
        continuous_scaled = loaded_scaler.transform(continuous_df)
        
        # Combine continuous scaled features with discrete data
        # Ensure the discrete data order matches what the model expects
        input_data = np.hstack((np.array(discrete_data).reshape(1, -1), continuous_scaled))
        
        # Make prediction
        prediction = loaded_model.predict(input_data)
        result = "Obese" if prediction[0] == 1 else "Non-obese"
        st.success(f"Predicted Obesity: {result}")
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
