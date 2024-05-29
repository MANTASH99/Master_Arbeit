from sklearn.model_selection import train_test_split
import numpy as np
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

# Define alpha values to try
alphas = np.logspace(-3, 3, 7)  # Example: Try 7 values ranging from 10^-3 to 10^3

# Initialize lists to store results
mae_scores = []
r2_scores = []

# Iterate over each alpha value
for alpha in alphas:
    # Create Ridge regression model with the current alpha value
    ridge_model = Ridge(alpha=alpha)

    # Perform cross-validation to evaluate the model with 5-fold CV
    cv_mae_scores = -cross_val_score(ridge_model, input_train, output_train.ravel(), cv=5,
                                     scoring='neg_mean_absolute_error')
    cv_r2_scores = cross_val_score(ridge_model, input_train, output_train.ravel(), cv=5, scoring='r2')

    # Record the mean MAE and R2 scores from cross-validation
    mae_scores.append(np.mean(cv_mae_scores))
    r2_scores.append(np.mean(cv_r2_scores))

# Find the index of the alpha value with the best MAE score
best_alpha_index = np.argmin(mae_scores)
best_alpha = alphas[best_alpha_index]

# Fit the final Ridge regression model with the best alpha
final_ridge_model = Ridge(alpha=best_alpha)
final_ridge_model.fit(input_train, output_train.ravel())

# Make predictions on the test set using the final model
output_pred_ridge = final_ridge_model.predict(input_test)

# Calculate Mean Absolute Error (MAE) loss and R2 score on the test set
mae_loss_ridge = mean_absolute_error(output_test, output_pred_ridge)
r2_ridge = r2_score(output_test, output_pred_ridge)

# Print the best alpha and corresponding MAE and R2 scores
print("Best alpha:", best_alpha)
print("Best MAE Loss:", mae_scores[best_alpha_index])
print("Best R2 Score:", r2_scores[best_alpha_index])

# Print the MAE and R2 score of the final model on the test set
print("Final Ridge Model MAE Loss:", mae_loss_ridge)
print("Final Ridge Model R2 Score:", r2_ridge)