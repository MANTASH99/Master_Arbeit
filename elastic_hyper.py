from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import ElasticNet

from sklearn.metrics import r2_score , mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import torch
from sklearn.preprocessing import MinMaxScaler
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
alphas = np.logspace(-3, 3, 7)  # Try 7 values ranging from 1e-3 to 1e3
l1_ratios = np.linspace(0.1, 0.9, 5)  # Try 5 different L1 ratios from 0.1 to 0.9

# Lists to store scores
mae_scores = []
r2_scores = []

# Loop over hyperparameters
for alpha in alphas:
    for l1_ratio in l1_ratios:
        # Create ElasticNet model with current hyperparameters
        elasticnet_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

        # Calculate cross-validation scores
        cv_mae_scores = -cross_val_score(elasticnet_model, input_train, output_train.ravel(),
                                         cv=5, scoring='neg_mean_absolute_error')
        cv_r2_scores = cross_val_score(elasticnet_model, input_train, output_train.ravel(),
                                       cv=5, scoring='r2')

        # Append mean scores to lists
        mae_scores.append(np.mean(cv_mae_scores))
        r2_scores.append(np.mean(cv_r2_scores))

# Find the best hyperparameters
best_alpha_index, best_l1_ratio_index = np.unravel_index(np.argmin(mae_scores), (len(alphas), len(l1_ratios)))
best_alpha = alphas[best_alpha_index]
best_l1_ratio = l1_ratios[best_l1_ratio_index]

print("Best Hyperparameters:")
print("Alpha:", best_alpha)
print("L1 Ratio:", best_l1_ratio)