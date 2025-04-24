import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib

# -------------------------------------------
# 🎯 Load the Trained Model
# -------------------------------------------
model_path = "xgb_brick_strength_model.pkl"  # Update with the correct model path
model = joblib.load(model_path)

# -------------------------------------------
# 🎨 Streamlit UI Configuration
# -------------------------------------------
st.title("🏗️ Brick Compressive Strength Prediction UI")
st.write("Enter the brick composition parameters below to predict the compressive strength.")

# -------------------------------------------
# 📥 Define Input Fields for User
# -------------------------------------------
cement = st.number_input("Cement (%)", min_value=0.0, max_value=100.0, value=50.0)
iron_ore = st.number_input("Iron Ore (%)", min_value=0.0, max_value=100.0, value=20.0)
fine_aggregate = st.number_input("Fine Aggregate (%)", min_value=0.0, max_value=100.0, value=30.0)
wfs = st.number_input("WFS (%)", min_value=0.0, max_value=100.0, value=10.0)
load = st.number_input("Load (kg)", min_value=0.0, max_value=100.0, value=5.0)
area = st.number_input("Area (cm²)", min_value=0.0, max_value=1000.0, value=50.0)

# -------------------------------------------
# 📊 Collect Input Data into Dictionary
# -------------------------------------------
input_data = {
    "Cement %": cement,
    "Iron Ore %": iron_ore,
    "Fine Aggregate %": fine_aggregate,
    "WFS %": wfs,
    "Load": load,
    "Area": area,
}

# -------------------------------------------
# 📊 Convert Input to DataFrame
# -------------------------------------------
input_df = pd.DataFrame([input_data])

# -------------------------------------------
# 🚀 Predict Button and Model Prediction
# -------------------------------------------
if st.button("🔮 Predict Strength"):
    try:
        # Make Prediction
        predicted_strength = model.predict(input_df)[0]
        st.success(f"✅ Predicted Compressive Strength: **{predicted_strength:.2f} N/mm²**")
    except Exception as e:
        st.error(f"❌ Error occurred while predicting: {e}")

# -------------------------------------------
# 📚 Display Limitations of the Model
# -------------------------------------------
st.write("### ⚠️ Model Limitations")
st.info(
    """
- Model trained on data with specific material limits. Values beyond the limits may reduce prediction accuracy.
- Ensure correct proportions of inputs for optimal prediction results.
- Always validate predictions with physical testing before making real-world decisions.
"""
)
