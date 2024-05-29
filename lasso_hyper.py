from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor ,GradientBoostingRegressor
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
alphas = np.logspace(-3, 3, 7)  # Example: Try 7 values ranging from 10^-3 to 10^3

# Initialize lists to store results
mae_scores = []
r2_scores = []

# Iterate over each alpha value
for alpha in alphas:
    # Create Lasso regression model with the current alpha value
    lasso_model = Lasso(alpha=alpha,max_iter=10000)

    # Perform cross-validation to evaluate the model with 5-fold CV
    cv_mae_scores = -cross_val_score(lasso_model, input_train, output_train.ravel(), cv=5,
                                     scoring='neg_mean_absolute_error')
    cv_r2_scores = cross_val_score(lasso_model, input_train, output_train.ravel(), cv=5, scoring='r2')

    # Record the mean MAE and R2 scores from cross-validation
    mae_scores.append(np.mean(cv_mae_scores))
    r2_scores.append(np.mean(cv_r2_scores))

# Find the index of the alpha value with the best MAE score
best_alpha_index = np.argmin(mae_scores)
best_alpha = alphas[best_alpha_index]

# Fit the final Lasso regression model with the best alpha
final_lasso_model = Lasso(alpha=best_alpha,max_iter=10000)
final_lasso_model.fit(input_train, output_train.ravel())

# Make predictions on the test set using the final model
output_pred_lasso = final_lasso_model.predict(input_test)

# Calculate Mean Absolute Error (MAE) loss and R2 score on the test set
mae_loss_lasso = mean_absolute_error(output_test, output_pred_lasso)
r2_lasso = r2_score(output_test, output_pred_lasso)

# Print the best alpha and corresponding MAE and R2 scores
print("Best alpha:", best_alpha)
print("Best MAE Loss:", mae_scores[best_alpha_index])
print("Best R2 Score:", r2_scores[best_alpha_index])

# Print the MAE and R2 score of the final model on the test set
print("Final Lasso Model MAE Loss:", mae_loss_lasso)
print("Final Lasso Model R2 Score:", r2_lasso)