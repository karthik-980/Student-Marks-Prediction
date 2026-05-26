import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("Student Marks Prediction")

st.write("Predict marks based on study hours")

# User input
hours = st.number_input("Enter Study Hours", min_value=0.0)

# Prediction button
if st.button("Predict"):
    import pandas as pd
    
    # Create a DataFrame with the same column name as training data
    input_data = pd.DataFrame({'hours': [hours]})
    prediction = model.predict(input_data)

    st.success(f"Predicted Marks: {prediction[0]:.2f}")