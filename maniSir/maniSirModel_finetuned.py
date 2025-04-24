import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the data
file_path = "D:/test/maniSir/test_data.csv"  # Replace with the actual path
  # Replace with your file path
data = pd.read_csv(file_path)

# Define the features and target
features = ["Cement %", "Iron Ore %", "Fine Aggregate %", "WFS %", "Load", "Area"]
target = "Compressive Strength (N/mm2)"

X = data[features]
y = data[target]

# Split the data into train, validation, and test sets (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Set up the XGBoost regressor
xgb_model = xgb.XGBRegressor()

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8],
    'gamma': [0, 0.1, 0.2]
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best model and its parameters
best_model = grid_search.best_estimator_

# Save the model to a file
joblib.dump(best_model, "xgb_finetuned_brick_strength_model.pkl")
print("Fine-tuned model saved to: xgb_finetuned_brick_strength_model.pkl")

# Make predictions on the validation and test sets
y_val_pred = best_model.predict(X_val)
y_test_pred = best_model.predict(X_test)

# Evaluate the model on validation set
val_mse = mean_squared_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)
print(f"Validation MSE: {val_mse}")
print(f"Validation R²: {val_r2}")

# Evaluate the model on test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Test MSE: {test_mse}")
print(f"Test R²: {test_r2}")
