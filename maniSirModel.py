import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Read the Excel data
file_path = "D:/test/maniSir/All_Brick_Strength_Data_For_ML.xlsx"
data = pd.read_excel(file_path)

# Selecting the features and target column
features = ["Cement %", "Iron Ore %", "Fine Aggregate %", "WFS %", "Load", "Area"]
target = "Compressive Strength (N/mm2)"

# Prepare the input features and target
X = data[features]
y = data[target]

# Split the dataset into training, validation, and test sets (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train the XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict on validation and test sets
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) for evaluation
val_mse = mean_squared_error(y_val, y_val_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Validation MSE: {val_mse}")
print(f"Test MSE: {test_mse}")

# Save the model to a file
model_filename = "D:/test/maniSir/xgb_brick_strength_model.pkl"
joblib.dump(model, model_filename)

# Optionally save the augmented dataset
X_train.to_csv("D:/test/maniSir/augmented_train.csv", index=False)
X_val.to_csv("D:/test/maniSir/augmented_val.csv", index=False)
X_test.to_csv("D:/test/maniSir/augmented_test.csv", index=False)

print(f"Model saved to: {model_filename}")
from sklearn.metrics import mean_squared_error, r2_score

# Assuming you have y_true (actual values) and y_pred (predicted values)
# For validation set
y_true_val = y_val  # Actual values for validation
y_pred_val = model.predict(X_val)  # Predicted values for validation
r2_val = r2_score(y_true_val, y_pred_val)

# For test set
y_true_test = y_test  # Actual values for test
y_pred_test = model.predict(X_test)  # Predicted values for test
r2_test = r2_score(y_true_test, y_pred_test)

# Print R² scores
print(f"Validation R²: {r2_val}")
print(f"Test R²: {r2_test}")
