from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import torch
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
data_frame = pd.read_csv('bi_all_1 - Kopie.csv', delimiter=';', encoding='iso-8859-1')
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

# Vectorize the conversion function
vectorized_convert = np.vectorize(convert_to_float)

# Convert string elements to floats
float_array = vectorized_convert(string_array)

output_data11 = data_frame.iloc[1:, -1].values
float_array_out = vectorized_convert(output_data11)

# Preprocess the data
input_data1 = float_array[1:, :]
input_data2 = torch.from_numpy(input_data1).float()
output_data2 = torch.from_numpy(float_array_out).view(-1, 1).float()

scaler = MinMaxScaler()
normalized_input = scaler.fit_transform(input_data2)
normalized_input1 = torch.from_numpy(normalized_input).float()

# Split the data into training and testing sets
input_train11, input_test, output_train11, output_test = train_test_split(normalized_input1, output_data2, test_size=0.2)
input_val, input_train, output_val, output_train = train_test_split(input_train11, output_train11, test_size=0.95)
has_nan = torch.isnan(input_train).any().item()

if has_nan:
    print("The tensor contains NaN values.")
else:
    print("The tensor does not contain NaN values.")

# Create and train the Bayesian Ridge regression model
bayesian_model = BayesianRidge()
bayesian_model.fit(input_train, output_train.ravel())

# Make predictions on the test set
output_pred_bayesian = bayesian_model.predict(input_test.reshape(-1, input_train.shape[1]))

# Calculate Mean Absolute Error (MAE) loss
mae_loss_bayesian = mean_absolute_error(output_test, output_pred_bayesian)
print("Bayesian Ridge Mean Absolute Error (MAE):", mae_loss_bayesian)

# Calculate R2 score
r2_bayesian = r2_score(output_test, output_pred_bayesian)
print("Bayesian Ridge R2 Score:", r2_bayesian)