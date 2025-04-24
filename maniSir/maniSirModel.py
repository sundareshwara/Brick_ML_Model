import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# -------------------------------------------
# 📥 Load the Dataset
# -------------------------------------------
file_path = "D:/test/maniSir/All_Brick_Strength_Data_For_ML.xlsx"
data = pd.read_excel(file_path)

# -------------------------------------------
# 🎯 Select Features and Target (NO Load & Area)
# -------------------------------------------
features = ["Cement %", "Iron Ore %", "Fine Aggregate %", "WFS %"]
target = "Compressive Strength (N/mm2)"

X = data[features]
y = data[target]

# -------------------------------------------
# 🔀 Split the Dataset
# -------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# -------------------------------------------
# 🚀 Train the XGBoost Model
# -------------------------------------------
model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# -------------------------------------------
# 📊 Evaluate the Model
# -------------------------------------------
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

val_mse = mean_squared_error(y_val, y_val_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
val_r2 = r2_score(y_val, y_val_pred)
test_r2 = r2_score(y_test, y_test_pred)

# -------------------------------------------
# 📢 Print Results
# -------------------------------------------
print(f"Validation MSE: {val_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Validation R²: {val_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# -------------------------------------------
# 💾 Save the Trained Model
# -------------------------------------------
model_filename = "D:/test/maniSir/xgb_brick_strength_model_v2.pkl"
joblib.dump(model, model_filename)
print(f"✅ Model saved to: {model_filename}")

# -------------------------------------------
# 💽 Save CSV files for each split
# -------------------------------------------
train_df = X_train.copy()
train_df["Target"] = y_train
val_df = X_val.copy()
val_df["Target"] = y_val
test_df = X_test.copy()
test_df["Target"] = y_test

train_df.to_csv("D:/test/maniSir/augmented_train_v2.csv", index=False)
val_df.to_csv("D:/test/maniSir/augmented_val_v2.csv", index=False)
test_df.to_csv("D:/test/maniSir/augmented_test_v2.csv", index=False)

print("📂 Train/Val/Test CSVs saved.")
