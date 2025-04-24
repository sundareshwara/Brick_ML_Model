import streamlit as st
import pandas as pd
import joblib

# 🎯 Load the Trained Model
model_path = "xgb_brick_strength_model.pkl"  # Update with the correct model path
model = joblib.load(model_path)

# 🎨 Streamlit UI Configuration
st.title("🏗️ Brick Compressive Strength Prediction UI")
st.write("Enter the brick composition parameters below to predict the compressive strength.")

# 📥 Define Input Fields for User
cement = st.number_input("Cement (%)", min_value=0.0, max_value=100.0, value=50.0)
iron_ore = st.number_input("Iron Ore (%)", min_value=0.0, max_value=100.0, value=20.0)
fine_aggregate = st.number_input("Fine Aggregate (%)", min_value=0.0, max_value=100.0, value=30.0)
wfs = st.number_input("WFS (%)", min_value=0.0, max_value=100.0, value=10.0)

# 🎯 Binder and Fines Percentages
binder_total = cement + iron_ore
fines_total = fine_aggregate + wfs

# 📝 Show Total Mix Percentages for Feedback
st.markdown(f"**🧱 Total Binder (Cement + Iron Ore):** `{binder_total:.2f}%` / 100%")
st.markdown(f"**🧱 Total Fines (Fine Aggregate + WFS):** `{fines_total:.2f}%` / 100%")

# 🚨 Ensure the percentages sum up correctly
if binder_total != 100.0:
    st.warning("⚠️ The sum of Cement (%) and Iron Ore (%) should be exactly 100%.")
if fines_total != 100.0:
    st.warning("⚠️ The sum of Fine Aggregate (%) and WFS (%) should be exactly 100%.")

# 📊 Collect Input Data into Dictionary
input_data = {
    "Cement %": cement,
    "Iron Ore %": iron_ore,
    "Fine Aggregate %": fine_aggregate,
    "WFS %": wfs,
}

# 📊 Convert Input to DataFrame
input_df = pd.DataFrame([input_data])

# 🚀 Predict Button and Model Prediction
if st.button("🔮 Predict Strength"):
    try:
        # Check if the inputs are valid (should sum to 100% for binder and fines)
        if binder_total == 100.0 and fines_total == 100.0:
            # Make Prediction
            predicted_strength = model.predict(input_df)[0]
            st.success(f"✅ Predicted Compressive Strength: **{predicted_strength:.2f} N/mm²**")
        else:
            st.error("❌ Please make sure both the binder and fines percentages sum to 100%.")
    except Exception as e:
        st.error(f"❌ Error occurred while predicting: {e}")

# 📚 Display Model Limitations
st.write("### ⚠️ Model Limitations")
st.info(
    """
- Model trained on data with specific material limits. Values beyond the limits may reduce prediction accuracy.
- Ensure correct proportions of inputs for optimal prediction results.
- Always validate predictions with physical testing before making real-world decisions.
"""
)
