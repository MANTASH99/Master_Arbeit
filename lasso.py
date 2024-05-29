from sklearn.model_selection import train_test_split
import numpy as np
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
    lasso_model = Lasso(alpha=0.1)  # Adjust alpha as needed
    lasso_model.fit(input_train, output_train)

    # Predict with Lasso Regression model
    lasso_predictions = lasso_model.predict(input_test)

    # Calculate MAE for Lasso Regression
    lasso_mae = mean_absolute_error(output_test, lasso_predictions)
    print(f"Lasso Regression MAE: {lasso_mae}")

    # Calculate R2 score for Lasso Regression
    lasso_r2 = r2_score(output_test, lasso_predictions)
    print(f"Lasso Regression R2 Score: {lasso_r2}")

    # Visualize actual vs. predicted values for Lasso Regression
    plt.figure(figsize=(10, 6))
    plt.scatter(output_test, lasso_predictions, c='red', label='WRS65 vs. Predicted WRS65')
    plt.xlabel('WRS65')
    plt.ylabel('Predicted WRS65')
    plt.title('Lasso Regression')
    plt.legend()
    plt.savefig('lass.png')
    plt.show()