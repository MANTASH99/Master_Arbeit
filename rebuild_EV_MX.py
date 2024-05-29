
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression , LogisticRegression ,Ridge ,Lasso ,ElasticNet

from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

data_frame = pd.read_csv('bi_all_1 - Kopie.csv' , delimiter=';')
string_array = data_frame.iloc[:, :-1].values.astype(str)
ddata_frame = pd.read_csv('bi_all_1 - Kopie.csv', delimiter= ';' , encoding='iso-8859-1')

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
# Create a Polynomial Regression pipeline
degree = 2  # Degree of the polynomial
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(input_train)
X_test_poly = poly_features.transform(input_test)

# Create a polynomial regression model
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, output_train)

# Make predictions
y_train_pred = poly_reg.predict(X_train_poly)
y_test_pred = poly_reg.predict(X_test_poly)

# Calculate training and testing errors
train_error = mean_squared_error(output_train, y_train_pred)
test_error = mean_squared_error(output_test, y_test_pred)

# Plot the results
plt.scatter(input_train, output_train, color='blue', label='Training Data')
plt.scatter(input_test, output_test, color='red', label='Testing Data')
plt.plot(input_train, y_train_pred, color='green', label=f'Polynomial Regression (Degree {degree})')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.show()

print(f'Training Mean Squared Error: {train_error:.2f}')
print(f'Testing Mean Squared Error: {test_error:.2f}')