import torch
import torch.nn as nn
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score
# data_frame = pd.read_csv('EV_max_edit.csv', delimiter=';')
data_frame = pd.read_csv('numbers.csv', delimiter= ';' , encoding='iso-8859-1')

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
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size ,drop_prob ):
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
        # out = self.drop_out(out)
        #

        # # # # out = self.drop_out(out)
        # # #
        # # #
        # # # # # out = self.drop_out(out)
        # # # #
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # # # #
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # # #
        # # #
        # # #
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # # # # out = self.drop_out(out)
        # # # # #
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # # # # out = self.drop_out(out)
        # # # # #
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # #out = self.drop_out(out)
        # out = self.fc2(out)
        # out = self.bn2(out)
        # out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #
        #
        #
        #
        # # out = self.drop_out(out)
        #
        #
        # #
        #

        out = self.fc3(out)
        # out = self.cliped_tensor(out)# use it at the end , not trainig
        return out
#7 layers 390 jidden size 500 rpoch for every thoing was good enough
test=12
# while test > 10:

model = NeuralNetwork(input_size=input_train.shape[1], hidden_size =300, drop_prob=0.1)
print((input_train.shape[1]))
baseline_model = nn.Linear(1, 1)
# best 5 7 layer 400 hidien sise

# Define the loss function
criterion = nn.L1Loss()
# Define the optimizer and learning rate
# weight_decay=0.01 ,
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.01)
num_epochs = 300
model.train()
val_losses = []
train_losses = []
mae_losses = []
# Implement the training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(input_train)
    loss = criterion(outputs, output_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())


    # def clip_predictions(predictions):
    #     clipped = torch.clamp(predictions, min=-1e15, max=1e15)
    #     return clipped

        # Calculate and append validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(input_test)
        val_mae = mean_absolute_error(val_outputs.numpy(), output_test.numpy())
        mae_losses.append(val_mae)

        val_loss = criterion(val_outputs, output_test)
        val_losses.append(val_loss.item())
        val_r2 = r2_score(output_test.numpy(), val_outputs.numpy())
        print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item()}, val_loss: {val_loss.item()}, val_r2: {val_r2}")

    model.train()
        #Print and monitor the loss during training
        # print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item()} , val_loss: {val_loss.item()}")

    # Evaluate the model on the test set
save_weight_matrix_path = "/Users/mahdimantash/Downloads/pathes/hearingloss_application.pth"
torch.save(model.state_dict(),save_weight_matrix_path)
def cliped_tensor1 (x):
        clipped = torch.clamp(x ,max=100 )
        return clipped
model.eval()
with torch.no_grad():
    test_outputs1 = model(input_test)
    test_outputs = cliped_tensor1(test_outputs1)
    test_loss = criterion(test_outputs, output_test)
    test = test_loss
    test_r2 = r2_score(output_test.numpy(), test_outputs.numpy())
    print(f"Test L1 Loss: {test_loss.item()},Test R2 Score: {test_r2}")

#, Test R2 Score: {test_r2}"
        # print(f"Test L1 Loss: {test_loss.item()}")
    #print(test_outputs)
    #print(output_test)
    ###ploott  tru vs predicted values
test_predictions = test_outputs.detach().numpy()
test_labels = output_test.numpy()


# plt.figure(figsize=(10, 6))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Learning Curve')
# plt.legend()
# plt.show()
#
#





# plt.figure(figsize=(8, 6))
# plt.scatter(test_labels, test_predictions, c='red', alpha=0.6)
# plt.xlabel("EV_65")
# plt.ylabel("Predicted_EV_65")
# plt.title("5_best_frequencies_HG_OWN")
# plt.grid(True)
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
#plt.plot(mae_losses, label='MAE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve 7 layers')
plt.legend()
plt.savefig('loss7.png')
plt.show()








