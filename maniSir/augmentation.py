import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib

# Load original Excel data
data = pd.read_excel('All_Brick_Strength_Data_For_ML.xlsx')

# Select input and output columns
features = ['Cement %', 'Iron Ore %', 'Fine Aggregate %', 'WFS %']
target = 'Compressive Strength (N/mm2)'

X = data[features]
y = data[target]

# Train XGBoost model on original data
model = XGBRegressor()
model.fit(X, y)

# Save model for reuse
joblib.dump(model, 'brick_strength_model.pkl')

# Data augmentation
desired_count = 1000
augmented_data = []

while len(augmented_data) < desired_count:
    # Pick a random original data point
    idx = random.randint(0, len(X) - 1)
    base = X.iloc[idx]

    # Add Gaussian noise (small variation)
    augmented_point = base + np.random.normal(0, 2, size=4)
    augmented_point = np.clip(augmented_point, 0, 100)  # Clamp to valid range

    # Predict compressive strength using the model
    strength = model.predict([augmented_point])[0]

    augmented_data.append(np.append(augmented_point, strength))

# Create DataFrame
columns = features + [target]
augmented_df = pd.DataFrame(augmented_data, columns=columns)

# Split into train (70%), val (15%), test (15%)
train_df, temp_df = train_test_split(augmented_df, test_size=0.30, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save to CSV files
train_df.to_csv('train_data.csv', index=False)
val_df.to_csv('val_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

print("✅ Synthetic data generation complete. Datasets saved:")
print(f" - Train: {len(train_df)} samples")
print(f" - Validation: {len(val_df)} samples")
print(f" - Test: {len(test_df)} samples")
