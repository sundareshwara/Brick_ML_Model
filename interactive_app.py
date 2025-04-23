import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle

# Load the trained model
model_path = 'D:/test/maniSir/xgb_brick_strength_model.pkl'
model = pickle.load(open(model_path, 'rb'))

# Function to make predictions
def predict_strength(data):
    data = pd.DataFrame([data], columns=['Cement %', 'Iron Ore %', 'Fine Aggregate %', 'WFS %', 'Load', 'Area'])
    prediction = model.predict(data)
    return prediction[0]

# Streamlit interface
st.title('Brick Strength Prediction')
st.write("Enter the details below to predict the compressive strength of the brick:")

# Input form
cement = st.number_input('Cement %', min_value=0.0, max_value=100.0, value=50.0)
iron_ore = st.number_input('Iron Ore %', min_value=0.0, max_value=100.0, value=20.0)
fine_aggregate = st.number_input('Fine Aggregate %', min_value=0.0, max_value=100.0, value=30.0)
wfs = st.number_input('WFS %', min_value=0.0, max_value=100.0, value=10.0)
load = st.number_input('Load', min_value=0.0, max_value=100.0, value=5.0)
area = st.number_input('Area', min_value=0.0, max_value=1000.0, value=50.0)

# Predict button
if st.button('Predict'):
    input_data = {
        'Cement %': cement,
        'Iron Ore %': iron_ore,
        'Fine Aggregate %': fine_aggregate,
        'WFS %': wfs,
        'Load': load,
        'Area': area
    }
    prediction = predict_strength(input_data)
    st.write(f"The predicted compressive strength is: {prediction:.2f} N/mm²")
