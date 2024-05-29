from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression , LogisticRegression ,Ridge ,Lasso ,ElasticNet
from sklearn.metrics import mean_absolute_error , r2_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
data_frame = pd.read_csv('bi_all_1 - Kopie.csv', delimiter= ';' , encoding='iso-8859-1')

string_array = data_frame.iloc[:, :-1].values.astype(str)


# Custom conversion function
def convert_to_float(value):
    try:
        # If it's already a float, return it as is
        if isinstance(value, float):
            return value
        # Otherwise, convert the string to float
        return float(value.replace(',', '.'))
    except ValueError:
        return np.nan

# print(data_frame)
# Vectorize the conversion function
vectorized_convert = np.vectorize(convert_to_float)

# Convert string elements to floats
float_array = vectorized_convert(string_array)

output_data11 = data_frame.iloc[1:, -1].values
# print(output_data11)


float_array_out = vectorized_convert(output_data11)
# output_data11 = vectorized_convert(data_frame.iloc[:, -1].values)
# Print the array with float elements


# input_data1 = float_array[:,0]
input_data1 = float_array[1: , :]

input_data2 = torch.from_numpy(input_data1).float()

# output_data1 = output_data1.astype(float)
output_data2 = torch.from_numpy(float_array_out).view(-1, 1).float()
# print(output_data2)
# scale the data between 0 and 1
scaler = MinMaxScaler()
normalized_input = scaler.fit_transform(input_data2)
normalized_input1 = torch.from_numpy(normalized_input).float()

input_train11, input_test, output_train11, output_test = train_test_split(normalized_input1, output_data2, test_size=0.2)
input_val, input_train, output_val, output_train = train_test_split(input_train11, output_train11, test_size=0.95)
has_nan = torch.isnan(input_train).any().item()

if has_nan:
    print("The tensor contains NaN values.")
else:
    print("The tensor does not contain NaN values.")
# Linear Regression

# Linear Regression
n_estimators = 100  # Number of trees in the forest
random_forest_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
random_forest_model.fit(input_train, output_train)

# Make predictions on the test set
output_pred_random_forest = random_forest_model.predict(input_test)

# Calculate Mean Absolute Error (MAE) loss
mae_loss_random_forest = mean_absolute_error(output_test, output_pred_random_forest)
print("Random Forest Regression - Mean Absolute Error (MAE):", mae_loss_random_forest)

# Calculate R² score
r2_random_forest = r2_score(output_test, output_pred_random_forest)
print("Random Forest Regression - R² Score:", r2_random_forest)

# Plot actual vs predicted values
feature_importances = random_forest_model.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': data_frame.columns[:-1], 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("Feature Importances:")
print(feature_importance_df)

# Set a threshold for feature importance
threshold = 0.001  # Adjust the threshold as needed

# Select features based on the threshold importance value
selected_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]['Feature'].tolist()
print("Selected Features:")
print(selected_features)

# Retain only selected features in the input data


# Train the Random Forest model with selected features
random_forest_model_selected = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
random_forest_model_selected.fit(input_train, output_train)

# Make predictions on the test set with selected features
output_pred_random_forest_selected = random_forest_model_selected.predict(input_test)

# Calculate Mean Absolute Error (MAE) loss with selected features
mae_loss_random_forest_selected = mean_absolute_error(output_test, output_pred_random_forest_selected)
print("Random Forest Regression with Selected Features - Mean Absolute Error (MAE):", mae_loss_random_forest_selected)

# Calculate R² score with selected features
r2_random_forest_selected = r2_score(output_test, output_pred_random_forest_selected)
print("Random Forest Regression with Selected Features - R² Score:", r2_random_forest_selected)

# Plot actual vs predicted values with selected features
# Add plotting code here

