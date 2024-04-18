import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
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

# Define features (X) and target variable (y) for MEDV
X_MEDV = data.drop(columns=['MEDV'])
y_MEDV = data['MEDV']

# Define features (X) and target variable (y) for CRIM
X_CRIM = data.drop(columns=['CRIM'])
y_CRIM = data['CRIM']

# Define features (X) and target variable (y) for CHAS
X_CHAS = data.drop(columns=['CHAS'])
y_CHAS = data['CHAS']

# Define features (X) and target variable (y) for ZN
X_ZN = data.drop(columns=['ZN'])
y_ZN = data['ZN']

# Define features (X) and target variable (y) for RM
X_RM = data.drop(columns=['RM'])
y_RM = data['RM']

# Split the data into training and test sets for MEDV
X_train_MEDV, X_test_MEDV, y_train_MEDV, y_test_MEDV = train_test_split(X_MEDV, y_MEDV, test_size=0.5, random_state=42)

# Split the data into training and test sets for CRIM
X_train_CRIM, X_test_CRIM, y_train_CRIM, y_test_CRIM = train_test_split(X_CRIM, y_CRIM, test_size=0.5, random_state=42)

# Split the data into training and test sets for CHAS
X_train_CHAS, X_test_CHAS, y_train_CHAS, y_test_CHAS = train_test_split(X_CHAS, y_CHAS, test_size=0.5, random_state=42)

# Split the data into training and test sets for ZN
X_train_ZN, X_test_ZN, y_train_ZN, y_test_ZN = train_test_split(X_ZN, y_ZN, test_size=0.5, random_state=42)

# Split the data into training and test sets for RM
X_train_RM, X_test_RM, y_train_RM, y_test_RM = train_test_split(X_RM, y_RM, test_size=0.5, random_state=42)

# Generate polynomial features for MEDV
poly_MEDV = PolynomialFeatures(degree=2)
X_train_poly_MEDV = poly_MEDV.fit_transform(X_train_MEDV)
X_test_poly_MEDV = poly_MEDV.transform(X_test_MEDV)

# Generate polynomial features for CRIM
poly_CRIM = PolynomialFeatures(degree=2)
X_train_poly_CRIM = poly_CRIM.fit_transform(X_train_CRIM)
X_test_poly_CRIM = poly_CRIM.transform(X_test_CRIM)

# Generate polynomial features for CHAS
poly_CHAS = PolynomialFeatures(degree=2)
X_train_poly_CHAS = poly_CHAS.fit_transform(X_train_CHAS)
X_test_poly_CHAS = poly_CHAS.transform(X_test_CHAS)

# Generate polynomial features for ZN
poly_ZN = PolynomialFeatures(degree=2)
X_train_poly_ZN = poly_ZN.fit_transform(X_train_ZN)
X_test_poly_ZN = poly_ZN.transform(X_test_ZN)

# Generate polynomial features for RM
poly_RM = PolynomialFeatures(degree=2)
X_train_poly_RM = poly_RM.fit_transform(X_train_RM)
X_test_poly_RM = poly_RM.transform(X_test_RM)

# Feature Scaling for MEDV
scaler_MEDV = StandardScaler()
X_train_scaled_MEDV = scaler_MEDV.fit_transform(X_train_poly_MEDV)
X_test_scaled_MEDV = scaler_MEDV.transform(X_test_poly_MEDV)

# Feature Scaling for CRIM
scaler_CRIM = StandardScaler()
X_train_scaled_CRIM = scaler_CRIM.fit_transform(X_train_poly_CRIM)
X_test_scaled_CRIM = scaler_CRIM.transform(X_test_poly_CRIM)

# Feature Scaling for CHAS
scaler_CHAS = StandardScaler()
X_train_scaled_CHAS = scaler_CHAS.fit_transform(X_train_poly_CHAS)
X_test_scaled_CHAS = scaler_CHAS.transform(X_test_poly_CHAS)

# Feature Scaling for ZN
scaler_ZN = StandardScaler()
X_train_scaled_ZN = scaler_ZN.fit_transform(X_train_poly_ZN)
X_test_scaled_ZN = scaler_ZN.transform(X_test_poly_ZN)

# Feature Scaling for RM
scaler_RM = StandardScaler()
X_train_scaled_RM = scaler_RM.fit_transform(X_train_poly_RM)
X_test_scaled_RM = scaler_RM.transform(X_test_poly_RM)

# Initialize the Elastic Net CV model for MEDV
elastic_net_model_MEDV = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], alphas=[0.1, 0.5, 1.0, 3.0, 5.0, 10.0], max_iter=5000, cv=5)

# Initialize the Elastic Net CV model for CRIM
elastic_net_model_CRIM = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], alphas=[0.1, 0.5, 1.0, 3.0, 5.0, 10.0], max_iter=5000, cv=5)

# Initialize the Elastic Net CV model for CHAS
elastic_net_model_CHAS = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], alphas=[0.1, 0.5, 1.0, 3.0, 5.0, 10.0], max_iter=5000, cv=5)

# Initialize the Elastic Net CV model for ZN
elastic_net_model_ZN = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], alphas=[0.1, 0.5, 1.0, 3.0, 5.0, 10.0], max_iter=5000, cv=5)

# Initialize the Elastic Net CV model for RM
elastic_net_model_RM = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], alphas=[0.1, 0.5, 1.0, 3.0, 5.0, 10.0], max_iter=5000, cv=5)

# Fit the Elastic Net CV model to the MEDV data
elastic_net_model_MEDV.fit(X_train_scaled_MEDV, y_train_MEDV)

# Fit the Elastic Net CV model to the CRIM data
elastic_net_model_CRIM.fit(X_train_scaled_CRIM, y_train_CRIM)

# Fit the Elastic Net CV model to the CHAS data
elastic_net_model_CHAS.fit(X_train_scaled_CHAS, y_train_CHAS)

# Fit the Elastic Net CV model to the ZN data
elastic_net_model_ZN.fit(X_train_scaled_ZN, y_train_ZN)

# Fit the Elastic Net CV model to the RM data
elastic_net_model_RM.fit(X_train_scaled_RM, y_train_RM)

# Make predictions using the best model for MEDV
y_pred_test_elastic_net_MEDV = elastic_net_model_MEDV.predict(X_test_scaled_MEDV)

# Make predictions using the best model for CRIM
y_pred_test_elastic_net_CRIM = elastic_net_model_CRIM.predict(X_test_scaled_CRIM)

# Make predictions using the best model for CHAS
y_pred_test_elastic_net_CHAS = elastic_net_model_CHAS.predict(X_test_scaled_CHAS)

# Make predictions using the best model for ZN
y_pred_test_elastic_net_ZN = elastic_net_model_ZN.predict(X_test_scaled_ZN)

# Make predictions using the best model for RM
y_pred_test_elastic_net_RM = elastic_net_model_RM.predict(X_test_scaled_RM)

# Evaluate the best model for MEDV
mse_test_elastic_net_MEDV = mean_squared_error(y_test_MEDV, y_pred_test_elastic_net_MEDV)
r_squared_test_elastic_net_MEDV = r2_score(y_test_MEDV, y_pred_test_elastic_net_MEDV)

# Evaluate the best model for CRIM
mse_test_elastic_net_CRIM = mean_squared_error(y_test_CRIM, y_pred_test_elastic_net_CRIM)
r_squared_test_elastic_net_CRIM = r2_score(y_test_CRIM, y_pred_test_elastic_net_CRIM)

# Evaluate the best model for CHAS
mse_test_elastic_net_CHAS = mean_squared_error(y_test_CHAS, y_pred_test_elastic_net_CHAS)
r_squared_test_elastic_net_CHAS = r2_score(y_test_CHAS, y_pred_test_elastic_net_CHAS)

# Evaluate the best model for ZN
mse_test_elastic_net_ZN = mean_squared_error(y_test_ZN, y_pred_test_elastic_net_ZN)
r_squared_test_elastic_net_ZN = r2_score(y_test_ZN, y_pred_test_elastic_net_ZN)

# Evaluate the best model for RM
mse_test_elastic_net_RM = mean_squared_error(y_test_RM, y_pred_test_elastic_net_RM)
r_squared_test_elastic_net_RM = r2_score(y_test_RM, y_pred_test_elastic_net_RM)

# Output the results for MEDV
print("Best Alpha for MEDV:", elastic_net_model_MEDV.alpha_)
print("Best L1 Ratio for MEDV:", elastic_net_model_MEDV.l1_ratio_)
print("Mean Squared Error (MSE) on Test Data for MEDV:", mse_test_elastic_net_MEDV)
print("R-squared on Test Data for MEDV:", r_squared_test_elastic_net_MEDV)

# Output the results for CRIM
print("\nBest Alpha for CRIM:", elastic_net_model_CRIM.alpha_)
print("Best L1 Ratio for CRIM:", elastic_net_model_CRIM.l1_ratio_)
print("Mean Squared Error (MSE) on Test Data for CRIM:", mse_test_elastic_net_CRIM)
print("R-squared on Test Data for CRIM:", r_squared_test_elastic_net_CRIM)

# Output the results for CHAS
print("\nBest Alpha for CHAS:", elastic_net_model_CHAS.alpha_)
print("Best L1 Ratio for CHAS:", elastic_net_model_CHAS.l1_ratio_)
print("Mean Squared Error (MSE) on Test Data for CHAS:", mse_test_elastic_net_CHAS)
print("R-squared on Test Data for CHAS:", r_squared_test_elastic_net_CHAS)

# Output the results for ZN
print("\nBest Alpha for ZN:", elastic_net_model_ZN.alpha_)
print("Best L1 Ratio for ZN:", elastic_net_model_ZN.l1_ratio_)
print("Mean Squared Error (MSE) on Test Data for ZN:", mse_test_elastic_net_ZN)
print("R-squared on Test Data for ZN:", r_squared_test_elastic_net_ZN)

# Output the results for ZN
print("\nBest Alpha for ZN:", elastic_net_model_ZN.alpha_)
print("Best L1 Ratio for ZN:", elastic_net_model_ZN.l1_ratio_)
print("Mean Squared Error (MSE) on Test Data for ZN:", mse_test_elastic_net_ZN)
print("R-squared on Test Data for ZN:", r_squared_test_elastic_net_ZN)

# Output the results for RM
print("\nBest Alpha for RM:", elastic_net_model_RM.alpha_)
print("Best L1 Ratio for RM:", elastic_net_model_RM.l1_ratio_)
print("Mean Squared Error (MSE) on Test Data for RM:", mse_test_elastic_net_RM)
print("R-squared on Test Data for RM:", r_squared_test_elastic_net_RM)

# Cross-validation for MEDV
cv_scores_MEDV = cross_val_score(elastic_net_model_MEDV, X_train_scaled_MEDV, y_train_MEDV, cv=5, scoring='r2')
print("\nCross-Validation R-squared Scores for MEDV:", cv_scores_MEDV)
print("Mean Cross-Validation R-squared Score for MEDV:", np.mean(cv_scores_MEDV))

# Cross-validation for CRIM
cv_scores_CRIM = cross_val_score(elastic_net_model_CRIM, X_train_scaled_CRIM, y_train_CRIM, cv=5, scoring='r2')
print("\nCross-Validation R-squared Scores for CRIM:", cv_scores_CRIM)
print("Mean Cross-Validation R-squared Score for CRIM:", np.mean(cv_scores_CRIM))

# Cross-validation for CHAS
cv_scores_CHAS = cross_val_score(elastic_net_model_CHAS, X_train_scaled_CHAS, y_train_CHAS, cv=5, scoring='r2')
print("\nCross-Validation R-squared Scores for CHAS:", cv_scores_CHAS)
print("Mean Cross-Validation R-squared Score for CHAS:", np.mean(cv_scores_CHAS))

# Cross-validation for ZN
cv_scores_ZN = cross_val_score(elastic_net_model_ZN, X_train_scaled_ZN, y_train_ZN, cv=5, scoring='r2')
print("\nCross-Validation R-squared Scores for ZN:", cv_scores_ZN)
print("Mean Cross-Validation R-squared Score for ZN:", np.mean(cv_scores_ZN))

# Cross-validation for RM
cv_scores_RM = cross_val_score(elastic_net_model_RM, X_train_scaled_RM, y_train_RM, cv=5, scoring='r2')
print("\nCross-Validation R-squared Scores for RM:", cv_scores_RM)
print("Mean Cross-Validation R-squared Score for RM:", np.mean(cv_scores_RM))

# Calculate baseline prediction for MEDV
baseline_prediction_MEDV = np.mean(y_train_MEDV)
baseline_predictions_MEDV = np.full_like(y_test_MEDV, fill_value=baseline_prediction_MEDV)

# Calculate baseline prediction for CRIM
baseline_prediction_CRIM = np.mean(y_train_CRIM)
baseline_predictions_CRIM = np.full_like(y_test_CRIM, fill_value=baseline_prediction_CRIM)

# Calculate baseline prediction for CHAS
baseline_prediction_CHAS = np.mean(y_train_CHAS)
baseline_predictions_CHAS = np.full_like(y_test_CHAS, fill_value=baseline_prediction_CHAS)

# Calculate baseline prediction for ZN
baseline_prediction_ZN = np.mean(y_train_ZN)
baseline_predictions_ZN = np.full_like(y_test_ZN, fill_value=baseline_prediction_ZN)

# Calculate baseline prediction for ZN
baseline_prediction_ZN = np.mean(y_train_ZN)
baseline_predictions_ZN = np.full_like(y_test_ZN, fill_value=baseline_prediction_ZN)

# Calculate baseline prediction for RM
baseline_prediction_RM = np.mean(y_train_RM)
baseline_predictions_RM = np.full_like(y_test_RM, fill_value=baseline_prediction_RM)

# Evaluate baseline performance for MEDV
baseline_mse_MEDV = mean_squared_error(y_test_MEDV, baseline_predictions_MEDV)
baseline_r_squared_MEDV = r2_score(y_test_MEDV, baseline_predictions_MEDV)

# Evaluate baseline performance for CRIM
baseline_mse_CRIM = mean_squared_error(y_test_CRIM, baseline_predictions_CRIM)
baseline_r_squared_CRIM = r2_score(y_test_CRIM, baseline_predictions_CRIM)

# Evaluate baseline performance for CHAS
baseline_mse_CHAS = mean_squared_error(y_test_CHAS, baseline_predictions_CHAS)
baseline_r_squared_CHAS = r2_score(y_test_CHAS, baseline_predictions_CHAS)

# Evaluate baseline performance for ZN
baseline_mse_ZN = mean_squared_error(y_test_ZN, baseline_predictions_ZN)
baseline_r_squared_ZN = r2_score(y_test_ZN, baseline_predictions_ZN)

# Evaluate baseline performance for RM
baseline_mse_RM = mean_squared_error(y_test_RM, baseline_predictions_RM)
baseline_r_squared_RM = r2_score(y_test_RM, baseline_predictions_RM)

# Compare with the best model performance for MEDV
if mse_test_elastic_net_MEDV < baseline_mse_MEDV:
    print("\nBest Ridge model outperforms baseline for MEDV.")
else:
    print("\nBaseline outperforms best Ridge model for MEDV.")

# Compare with the best model performance for CRIM
if mse_test_elastic_net_CRIM < baseline_mse_CRIM:
    print("\nBest Ridge model outperforms baseline for CRIM.")
else:
    print("\nBaseline outperforms best Ridge model for CRIM.")

# Compare with the best model performance for CHAS
if mse_test_elastic_net_CHAS < baseline_mse_CHAS:
    print("\nBest Ridge model outperforms baseline for CHAS.")
else:
    print("\nBaseline outperforms best Ridge model for CHAS.")

# Compare with the best model performance for ZN
if mse_test_elastic_net_ZN < baseline_mse_ZN:
    print("\nBest Ridge model outperforms baseline for ZN.")
else:
    print("\nBaseline outperforms best Ridge model for ZN.")

# Compare with the best model performance for RM
if mse_test_elastic_net_RM < baseline_mse_RM:
    print("\nBest Ridge model outperforms baseline for RM.")
else:
    print("\nBaseline outperforms best Ridge model for RM.")

plt.figure(figsize=(15, 10))

# Plotting MEDV results
plt.scatter(y_test_MEDV, y_pred_test_elastic_net_MEDV, label='MEDV Predictions', color='blue')

# Plotting CRIM results
plt.scatter(y_test_CRIM, y_pred_test_elastic_net_CRIM, label='CRIM Predictions', color='red')

# Plotting CHAS results
plt.scatter(y_test_CHAS, y_pred_test_elastic_net_CHAS, label='CHAS Predictions', color='green')

# Plotting ZN results
plt.scatter(y_test_ZN, y_pred_test_elastic_net_ZN, label='ZN Predictions', color='orange')

# Plotting RM results
plt.scatter(y_test_RM, y_pred_test_elastic_net_RM, label='RM Predictions', color='purple')

# Plotting perfect predictions line
plt.plot([min(y_test_MEDV.min(), y_test_CRIM.min(), y_test_CHAS.min(), y_test_ZN.min(), y_test_RM.min()), 
          max(y_test_MEDV.max(), y_test_CRIM.max(), y_test_CHAS.max(), y_test_ZN.max(), y_test_RM.max())], 
         [min(y_test_MEDV.min(), y_test_CRIM.min(), y_test_CHAS.min(), y_test_ZN.min(), y_test_RM.min()), 
          max(y_test_MEDV.max(), y_test_CRIM.max(), y_test_CHAS.max(), y_test_ZN.max(), y_test_RM.max())], 
         color='black', linestyle='-', linewidth=1, label='Perfect predictions')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.title('Model Predictions on Test Data')
plt.show()
