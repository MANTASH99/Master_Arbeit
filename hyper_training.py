import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid





data_frame = pd.read_csv('dor_out_bi_all_12.csv', delimiter=';', encoding='iso-8859-1')
string_array = data_frame.iloc[:, :-1].values.astype(str)
# Custom conversion function
def convert_to_float(value):
    try:
        return float(value.replace(',', '.'))
    except ValueError:
        return np.nan

# Vectorize the conversion function
vectorized_convert = np.vectorize(convert_to_float)

# Convert string elements to floats
float_array = vectorized_convert(string_array)
output_data11 = data_frame.iloc[1:, -1].values
float_array_out = vectorized_convert(output_data11)
# output_data11 = vectorized_convert(data_frame.iloc[:, -1].values)
# Print the array with float elements


# input_data1 = float_array[:,0]
input_data1 = float_array[1: , :]
input_data2 = torch.from_numpy(input_data1).float()
# output_data1 = output_data1.astype(float)
output_data2 = torch.from_numpy(float_array_out).view(-1, 1).float()
# scale the data between 0 and 1
scaler = MinMaxScaler()
normalized_input = scaler.fit_transform(input_data2)
normalized_input1 = torch.from_numpy(normalized_input).float()
normalized_input1_train = normalized_input1[0:400,:]
normalized_input1_test = normalized_input1[400:,:]
output_data2_train = output_data2[0:400]
output_data2_test = output_data2[400:]

# Split data into training and test sets
# input_train11, input_test, output_train11, output_test = train_test_split(normalized_input1, output_data, test_size=0.2)
# input_val, input_train, output_val, output_train = train_test_split(input_train11, output_train11, test_size=0.9)

# Define the neural network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.LeakyReLU()
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def cliped_tensor(self, x):
        clipped = torch.clamp(x, max=100)
        return clipped

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.cliped_tensor(out)
        return out
#grid search , basiran search , optuna .,
# Set hyperparameters and initialize results list
param_combinations = {
    'num_epochs': [500,600,700,800,900,1000,1100,1200, 1500,1700,2000],
    'hidden_size': [10, 20, 50,60,80,90,10],
    'learning_rate': [0.001, 0.01, 0.1],
}
results = []

# Number of folds for cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Loop over each hyperparameter combination
for params in ParameterGrid(param_combinations):
    # Cross-validation loop
    for fold, (train_index, val_index) in enumerate(kf.split(normalized_input1_train)):
        # Split the dataset into training and validation sets
        input_train, input_val = normalized_input1_train[train_index], normalized_input1_train[val_index]
        output_train, output_val = output_data2_train[train_index], output_data2_train[val_index]

        # Initialize model
        model = NeuralNetwork(input_size=76, hidden_size=params['hidden_size'])

        # Define the optimizer and learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.L1Loss()
        # Training loop
        for epoch in range(params['num_epochs']):
            model.train()
            optimizer.zero_grad()
            outputs = model(input_train)
            loss = criterion(outputs, output_train)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(input_val)
            val_loss = criterion(val_outputs, output_val)

        # Record the results for this combination and fold
        results.append({
            'params': params,
            'fold': fold,
            'validation_loss': val_loss.item(),
        })

# Find the best set of hyperparameters based on the minimum average validation loss
best_params = min(results, key=lambda x: x['validation_loss'])['params']

print("Best Hyperparameters:")
print(best_params)