import streamlit as st
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def train_model_with_smote(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    
    return model, scaler

# Load data and train model
@st.cache_resource
def load_or_train_model():
    try:
        model = joblib.load("model/Random Forest_model.pkl")
        scaler = joblib.load("model/scaler.pkl")
    except FileNotFoundError:
        # Train model if files don't exist
        data = pd.read_csv("your_data.csv")  # Replace with your data path
        X = data.drop('obesity', axis=1)
        y = data['obesity']
        model, scaler = train_model_with_smote(X, y)
        
        # Save model and scaler
        joblib.dump(model, "model/Random Forest_model.pkl")
        joblib.dump(scaler, "model/scaler.pkl")
    
    return model, scaler

# Load model and scaler
model, scaler = load_or_train_model()

# UI elements remain the same as previous code
# ... [Previous UI code remains unchanged]

if st.button("Predict Obesity"):
    try:
        continuous_df = pd.DataFrame([continuous_data], columns=continuous_fields.keys())
        continuous_scaled = scaler.transform(continuous_df)
        input_data = np.hstack((np.array(discrete_data).reshape(1, -1), continuous_scaled))
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
        result = "Obese" if prediction[0] == 1 else "Non-obese"
        probability = probabilities[0][1] if prediction[0] == 1 else probabilities[0][0]
        
        st.success(f"Predicted: {result} (Confidence: {probability:.2%})")
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
