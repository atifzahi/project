# Research Question: How does historical pesticide usage influence agricultural sustainability and crop productivity over time?

# ---------------------------------------------
# Final Year Project Code: Pesticide Usage Trends
# ---------------------------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
from google.colab import fil
# -----------------------------
# Step 1: Load and Inspect Dataset
# -----------------------------

print("Please upload your CSV file")
uploaded = files.upload()
pesticides = pd.read_csv(uploaded)

print("\nInitial Dataset Info:")
print(pesticides.info())
print("\nFirst 5 rows:")
print(pesticides.head())

# -----------------------------
# Step 2: Clean and Preprocess Data
# -----------------------------
print("\nCleaning and preprocessing data...")
pesticides.columns = pesticides.columns.str.strip().str.lower()
label_encoders = {}
for col in ['area', 'item', 'unit']:
    le = LabelEncoder()
    pesticides[col] = le.fit_transform(pesticides[col])
    label_encoders[col] = le

# -----------------------------
# Step 3: Feature and Target Selection
# -----------------------------
X = pesticides[['area', 'item', 'year']]
y = pesticides['value']

# -----------------------------
# Step 4: Train-Test Split
# -----------------------------
print("\nSplitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Step 5: Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Step 6: Model 1 - Linear Regression
# -----------------------------
print("\nTraining Linear Regression Model...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
print("\nLinear Regression Results:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_lr):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_lr):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_lr):.2f}")

# -----------------------------
# Step 7: Model 2 - Random Forest
# -----------------------------
print("\nTraining Random Forest Model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
print("\nRandom Forest Results:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_rf):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_rf):.2f}")

# Feature importance
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 6))
feature_importances.sort_values().plot(kind='barh')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.show()

# -----------------------------
# Step 8: Model 3 - XGBoost
# -----------------------------
print("\nTraining XGBoost Model...")
xgb_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
print("\nXGBoost Results:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_xgb):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_xgb):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_xgb):.2f}")

# -----------------------------
# Step 9: Model Comparison Summary
# -----------------------------
print("\nModel Comparison Summary:")
results = {
    "Linear Regression": [mean_absolute_error(y_test, y_pred_lr), mean_squared_error(y_test, y_pred_lr), r2_score(y_test, y_pred_lr)],
    "Random Forest": [mean_absolute_error(y_test, y_pred_rf), mean_squared_error(y_test, y_pred_rf), r2_score(y_test, y_pred_rf)],
    "XGBoost": [mean_absolute_error(y_test, y_pred_xgb), mean_squared_error(y_test, y_pred_xgb), r2_score(y_test, y_pred_xgb)]
}

comparison_df = pd.DataFrame(results, index=['MAE', 'MSE', 'R2']).T
print(comparison_df)

# Optional: Save results
comparison_df.to_csv("model_comparison_results.csv")
print("\nModel comparison saved as 'model_comparison_results.csv'")

