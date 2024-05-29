import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
data_frame = pd.read_csv('ev_max_full.csv' , delimiter=';')
string_array = data_frame.iloc[:, :-1].values.astype(str)
def convert_to_float(value):
    try:
        return float(value.replace(',', '.'))
    except ValueError:
        return np.nan

vectorized_convert = np.vectorize(convert_to_float)

# Convert string elements to floats
float_array = vectorized_convert(string_array)
#output_data11 = vectorized_convert(data_frame.iloc[:, -1].values)
# Print the array with float elements


input_data1 = float_array
output_data1 = data_frame.iloc[:, -1].values
input_data2 = torch.from_numpy(input_data1).float()
output_data2 = torch.from_numpy(output_data1).view(-1, 1).float()
# Vectorize the conversion function
# vectorized_convert = np.vectorize(convert_to_float)
# float_array = vectorized_convert(string_array)
# #input = float_array[:,0]
# output_data1 = data_frame.iloc[:, -1].values
# input_data2 = torch.from_numpy(float_array).float()
# output_data2 = torch.from_numpy(output_data1).view(-1, 1).float()
input_train1, input_test, output_train1, output_test = train_test_split(input_data2, output_data2, test_size=0.2)
input_train, input_val, output_train, output_val = train_test_split(input_train1, output_train1, test_size=0.1)
print("Training set shapes:", input_train.shape, output_train.shape)
print("Validation set shapes:", input_val.shape, output_val.shape)
print("Test set shapes:", input_test.shape, output_test.shape)



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
    def cliped_tensor(self , x):
        clipped = torch.clamp(x , max=100)
        return clipped
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.cliped_tensor(out)
        # out = self.fc2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        # out= self.fc3(out)

        return out



#76own
model = NeuralNetwork(input_size=14, hidden_size=80)
baseline_model = nn.Linear(1, 1)


# Define the loss function
criterion = nn.L1Loss()

# Define the optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
num_epochs = 3000
mean_value = torch.mean(output_train)

# Create a baseline model that predicts the mean value with the same number of elements as output_train
baseline_model = torch.full_like(output_train, mean_value)

# Calculate the Smooth L1 loss for the baseline model on training data
baseline_train_loss = criterion(baseline_model, output_train)

print("Baseline Training Smooth L1 Loss:", baseline_train_loss.item())
# implemnent a base line to seee if my loss is good or badd or wahtever you got it
# mean_value = torch.mean(output_train)
# base_line_train_pred = torch.full_like(output_train , mean_value)
# base_line_test_pred = torch.full_like(output_test , mean_value)
# mse_creterion = nn.MSELoss()
# base_line_train_pred = mse_creterion(base_line_train_pred , output_train)
# base_line_test_pred = mse_creterion(base_line_test_pred , output_test)
# print("Baseline Training MSE Loss:", base_line_train_pred.item())
# print("Baseline Validation MSE Loss:", base_line_test_pred.item())
model.train()
val_losses = []
train_losses = []


for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(input_train)
    loss = criterion(outputs, output_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

# nouurr was here
    model.eval()
    with torch.no_grad():
        val_outputs = model(input_val)
        val_loss = criterion(val_outputs, output_val)
        val_losses.append(val_loss.item())
    model.train()

    print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item()} , Validation Loss: {val_loss.item()} ")
#,Validation Loss: {val_loss.item()}
save_weight_matrix_path = r"C:\Users\Ackermann\OneDrive\Desktop\pathes.pth"
torch.save(model.state_dict(),save_weight_matrix_path)
model.eval()
with torch.no_grad():
    test_outputs = model(input_test)
    test_loss = criterion(test_outputs, output_test)
    print(f"Test Loss: {test_loss.item()}")
#print(test_outputs)
#print(output_test)
###ploott  tru vs predicted values
test_predictions = test_outputs.detach().numpy()
test_labels = output_test.numpy()


plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.show()
plt.figure(figsize=(8, 6))
plt.scatter(test_labels, test_predictions, c='blue', alpha=0.6)
plt.xlabel("True Labels")
plt.ylabel("Predicted Labels")
plt.title("True Labels vs. Predicted Labels")
plt.grid(True)
plt.savefig('Smoothl1_loss_NN.png')
plt.show()





#TO do list  : rename the own to rear , rear 65 100 beispielweise , hg-ohr , pta , 500 to 4000 hz

