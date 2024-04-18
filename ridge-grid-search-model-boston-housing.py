import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load housing file
data = pd.read_csv('housing.csv', header=None, sep='\s+')

# Define column names
# CRIM: Per capita crime rate by town
# ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUS: Proportion of non-retail business acres per town
# CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# NOX: Nitric oxides concentration (parts per 10 million)
# RM: Average number of rooms per dwelling
# AGE: Proportion of owner-occupied units built prior to 1940
# DIS: Weighted distances to five Boston employment centers
# RAD: Index of accessibility to radial highways
# TAX: Full-value property tax rate per $10,000
# PTRATIO: Pupil-teacher ratio by town
# B: 1000(Bk - 0.63)^2 where Bk is the proportion of Black individuals by town
# LSTAT: Percentage of lower status of the population
# MEDV: Median value of owner-occupied homes in $1000's
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Assign column names to the DataFrame
data.columns = column_names

# Display the DataFrame
#print(data)

# Define features (X) and target variable (y)
X = data.drop(columns=['MEDV'])
y = data['MEDV']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Generate polynomial features
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# Define the paramter grid for grid search
param_grid = {'alpha': [0.1, 0.5, 1.0, 3.0, 4.0, 5.0, 10.0]}

# Initialize the Ridge regression model
ridge_model = Ridge() 

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=ridge_model, param_grid=param_grid, cv=5, scoring='r2') 

# Fit the grid search to the scaled training data
grid_search.fit(X_train_scaled, y_train)

# Get the best alpha value
best_alpha = grid_search.best_params_['alpha']
print("Best Alpha: ", best_alpha)

# Get the best Ridge model
best_ridge_model = grid_search.best_estimator_

# Make predictions using the best model
y_pred_test_best = best_ridge_model.predict(X_test_scaled)

# Evaluate the best model
mse_test_best = mean_squared_error(y_test, y_pred_test_best)
r_squared_test_best = r2_score(y_test, y_pred_test_best)

# Cross-validation
cv_scores = cross_val_score(best_ridge_model, X_train_scaled, y_train, cv=5, scoring='r2')
print("Cross-Validation R-squared Scores:", cv_scores)
print("Mean Cross-Validation R-squared Score:", np.mean(cv_scores))

# Calculate baseline prediction
baseline_prediction = np.mean(y_train)
baseline_predictions = np.full_like(y_test, fill_value=baseline_prediction)

# Evaluate baseline performance
baseline_mse = mean_squared_error(y_test, baseline_predictions)
baseline_r_squared = r2_score(y_test, baseline_predictions)

# Compare with the best model performance
print("\nBaseline Mean Squared Error (MSE):", baseline_mse)
print("Baseline R-squared:", baseline_r_squared)
print("\nBest Ridge Regression Model with Scaled Features:")
print("Mean Squared Error (MSE) on Test Data:", mse_test_best)
print("R-squared on Test Data:", r_squared_test_best)

# Interpret the results
if mse_test_best < baseline_mse:
    print("Best Ridge model outperforms baseline.")
else:
    print("Baseline outperforms best Ridge model.")
    
# Visualize the results
plt.scatter(y_test, y_pred_test_best, label='Test data')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linestyle='-', linewidth=1, label='Perfect predictions')
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values (y_pred)')
plt.legend()
plt.title('Model Predictions on Test Data')
plt.show()