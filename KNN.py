from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import  KNeighborsRegressor
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
knn_model = KNeighborsRegressor(n_neighbors=5)

# Train the KNN model
knn_model.fit(input_train, output_train)

# Make predictions on the test set
output_pred_knn = knn_model.predict(input_test)

# Calculate Mean Absolute Error (MAE) loss
mae_loss_knn = mean_absolute_error(output_test, output_pred_knn)
print("KNN Mean Absolute Error (MAE):", mae_loss_knn)

# Calculate R2 score
r2_knn = r2_score(output_test, output_pred_knn)
print("KNN R2 Score:", r2_knn)

plt.figure(figsize=(8, 6))
plt.scatter(output_test, output_pred_knn, c='red', alpha=0.6, label='Predictions')
#plt.plot(output_test, output_test, c='blue', label='Perfect Predictions')  # Plot the perfect predictions line
plt.xlabel("WRS65")
plt.ylabel("Predicted WRS65")
plt.title("KNN prediction ")
plt.legend()
plt.grid(True)

plt.savefig('KNN11.png')
plt.show()