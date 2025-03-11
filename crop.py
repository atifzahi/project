# Research Question: How does historical pesticide usage influence agricultural sustainability and crop productivity over time?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from google.colab import files

# Upload dataset
uploaded = files.upload()

# Load pesticide dataset
file_name = next(iter(uploaded))
pesticides = pd.read_csv(file_name)

# Display basic info
print(pesticides.info())
print(pesticides.head())

# Standardize column names
pesticides.columns = pesticides.columns.str.strip().str.lower()

# Encode categorical variables
label_encoders = {}
for col in ['area', 'item', 'unit']:
    le = LabelEncoder()
    pesticides[col] = le.fit_transform(pesticides[col])
    label_encoders[col] = le

# Feature selection (Predicting pesticide usage trends)
X = pesticides[['area', 'item', 'year']]
y = pesticides['value']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MAE: {mae}, MSE: {mse}, R2: {r2}')

# Plot feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values().plot(kind='barh')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Pesticide Usage Prediction')
plt.show()
