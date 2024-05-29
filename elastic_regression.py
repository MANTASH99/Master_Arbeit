from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
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

# Prepare input and output data
input_data1 = float_array[1:, :]
output_data2 = float_array_out.reshape(-1, 1)

# Scale the input data between 0 and 1
scaler = MinMaxScaler()
normalized_input = scaler.fit_transform(input_data1)
normalized_input1 = torch.from_numpy(normalized_input).float()

# Split the data into training and testing sets
input_train, input_test, output_train, output_test = train_test_split(normalized_input1, output_data2, test_size=0.2)
input_val, input_train, output_val, output_train = train_test_split(input_train, output_train, test_size=0.95)

# Check for NaN values in the training data
has_nan = torch.isnan(input_train).any().item()

if has_nan:
    print("The tensor contains NaN values.")
else:
    print("The tensor does not contain NaN values.")

# Elastic Net Regression
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Adjust alpha and l1_ratio as needed
elastic_net.fit(input_train, output_train.ravel())
elastic_net_predictions = elastic_net.predict(input_test)

# Evaluate the model using Mean Absolute Error (MAE) and R2 score
mae = mean_absolute_error(output_test, elastic_net_predictions)
r2 = r2_score(output_test, elastic_net_predictions)
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)

# Visualize the predictions
plt.figure(figsize=(8, 6))
plt.scatter(output_test, elastic_net_predictions, c='red', alpha=0.6, label='Predictions')
#plt.plot(output_test, output_test, c='blue', label='Perfect Predictions')  # Plot the perfect predictions line
plt.xlabel("WRS65")
plt.ylabel("Predicted WRS65")
plt.title("Elastic Net Regression Predictions")
plt.legend()
plt.grid(True)
plt.savefig('elastic.png')
plt.show()