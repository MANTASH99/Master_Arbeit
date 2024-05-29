import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
# data_frame = pd.read_csv('EV_max_edit.csv', delimiter=';')
data_frame = pd.read_csv('bi_all_1 - Kopie.csv', delimiter=';',  encoding='iso-8859-1')

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

# input_train11, input_test, output_train11, output_test = train_test_split(normalized_input1, output_data2, test_size=0.2)
# input_val, input_train, output_val, output_train = train_test_split(input_train11, output_train11, test_size=0.9)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size,drop_prob):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.drop_out = nn.Dropout(drop_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.LeakyReLU()
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    def cliped_tensor (self,x):
        clipped = torch.clamp(x , max= 100 )
        return clipped
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop_out(out)
        # drop out layer , increase hidden and then drop out
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop_out(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop_out(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop_out(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop_out(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop_out(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop_out(out)

        out = self.fc3(out)
        # out = self.cliped_tensor(out)# use it at the end , not trainig
        return out



# model = NeuralNetwork(input_size=76, hidden_size=20)
baseline_model = nn.Linear(1, 1)


# Define the loss function
criterion = nn.L1Loss()

# Define the optimizer and learning rate
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
num_epochs = 400
hidden_size = 300
learning_rate = 0.001

# Number of folds for cross-validation
num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Results lists to store performance metrics for each fold
train_losses_per_fold = []
val_losses_per_fold = []
best_val_loss = float('inf')
best_weights = None
for fold, (train_index, val_index) in enumerate(kf.split(normalized_input1_train)):
    # Split the dataset into training and validation sets
    input_train, input_val = normalized_input1_train[train_index], normalized_input1_train[val_index]
    output_train, output_val = output_data2_train[train_index], output_data2_train[val_index]

    # Initialize model
    model = NeuralNetwork(input_size=394, hidden_size=hidden_size,drop_prob=0.3
                          )

    # Define the optimizer and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.01)
    # tune also weight decay
# Implement the training loop
    for epoch in range(num_epochs):
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

        # Record the results for this fold
        train_losses_per_fold.append(loss.item())
        val_losses_per_fold.append(val_loss.item())
    # print(f"Fold: {fold + 1}/{num_folds}, Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item()} , val_loss: {val_loss.item()}")

        # Save the weights after the training loop for each fold
    save_weight_matrix_path = f"C:/Users/Ackermann/OneDrive/Desktop/test_fold_{fold}.pth"
    torch.save(model.state_dict(), save_weight_matrix_path)

    # Check if the current fold has the best validation loss so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = model.state_dict()

    # Load the best weights for the final evaluation
    # model.load_state_dict(best_weights)
        # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(input_val)
        val_loss = criterion(val_outputs, output_val)

    # Record the results for this fold
    train_losses_per_fold.append(loss.item())
    val_losses_per_fold.append(val_loss.item())
    print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item()} , val_loss: {val_loss.item()}")
# Evaluate the model on the test set
# save_weight_matrix_path = r"C:\Users\Ackermann\OneDrive\Desktop\test.pth"
# torch.save(model.state_dict(),save_weight_matrix_path)
model.load_state_dict(best_weights)
model.eval()
with torch.no_grad():
    test_outputs = model(normalized_input1_test)
    test_loss = criterion(test_outputs, output_data2_test)
    print(f"Test L1 Loss: {test_loss.item()}")
#print(test_outputs)
#print(output_test)
###ploott  tru vs predicted values
test_predictions = test_outputs.detach().numpy()
test_labels = output_data2_test.numpy()

# Step 1: Create the scatter plot
plt.figure(figsize=(10, 6))
plt.plot(train_losses_per_fold, label='Training Loss')
plt.plot(val_losses_per_fold, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.show()
#
#
plt.figure(figsize=(8, 6))
plt.scatter(test_labels, test_predictions, c='blue', alpha=0.6)
plt.xlabel("True Labels")
plt.ylabel("Predicted Labels")
plt.title("True Labels vs. Predicted Labels")
plt.grid(True)
plt.savefig('NN_cleaned_cross')
plt.show()
