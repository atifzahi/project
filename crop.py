# Research Question: How does historical pesticide usage influence agricultural sustainability and crop productivity over time?

# ---------------------------------------------
# Final Year Project Code: Pesticide Usage Trends
# ---------------------------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from google.colab import files
import io

# Upload the dataset
uploaded = files.upload()
filename = list(uploaded.keys())[0]
content = uploaded[filename]

# Read the CSV into a pandas DataFrame
df = pd.read_csv(io.BytesIO(content))

# Data Exploration
print("Data Overview:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

# Visualizing Missing Values
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Dropping unnecessary columns (if any, like 'Unnamed: 0')
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# Handle missing values by filling with mean for numerical columns
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Feature Engineering
# Interaction term between 'pesticides_tonnes' and 'avg_temp' for example
df['pesticide_temp_interaction'] = df['pesticides_tonnes'] * df['avg_temp']

# Temporal trend: Adding difference from the median year for time-based trend extraction
df['year_diff'] = df['Year'] - df['Year'].median()

# Split data into features and target
X = df[['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'pesticide_temp_interaction', 'year_diff']]
y = df['hg/ha_yield']

# Scaling features (important for some models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
}

# Hyperparameter Tuning with GridSearchCV for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# GridSearchCV for Random Forest and Gradient Boosting
grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5)
grid_search_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb, cv=5)

# Fit the grid searches
grid_search_rf.fit(X_train, y_train)
grid_search_gb.fit(X_train, y_train)

best_rf = grid_search_rf.best_estimator_
best_gb = grid_search_gb.best_estimator_

# Model Stacking: Combine predictions from multiple models for improved performance
estimators = [
    ('lr', LinearRegression()),
    ('rf', best_rf),
    ('gb', best_gb)
]

stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stacking_model.fit(X_train, y_train)

# Make Predictions and Evaluate
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model.__class__.__name__} Performance:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")
    print("-" * 50)

# Evaluate all models
evaluate_model(best_rf, X_test, y_test)
evaluate_model(best_gb, X_test, y_test)
evaluate_model(stacking_model, X_test, y_test)

# Visualizing the predictions vs actual values for the stacked model
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=stacking_model.predict(X_test))
plt.xlabel('Actual Crop Yield')
plt.ylabel('Predicted Crop Yield')
plt.title('Actual vs Predicted Crop Yield (Stacked Model)')
plt.show()

# Residuals Plot for the Stacked Model
plt.figure(figsize=(10, 6))
sns.residplot(x=stacking_model.predict(X_test), y=y_test - stacking_model.predict(X_test), lowess=True, line_kws={'color': 'red'})
plt.xlabel('Predicted Crop Yield')
plt.ylabel('Residuals')
plt.title('Residuals Plot (Stacked Model)')
plt.show()

