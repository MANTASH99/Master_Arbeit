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
def predict_output():
    model = NeuralNetwork(input_train.shape[1], hidden_size =300, drop_prob=0.1)
    model.load_state_dict(torch.load("/Users/mahdimantash/Downloads/pathes/hearingloss_application.pth"))  # Load your trained model here

    # Get input value from entry field
    input_value = entry.get()
    #
    # # Convert input value to float (assuming it's numeric)
    input_value = float(input_value)
    #
    # # Convert input value to tensor
    input_tensor = torch.tensor([[input_value]], dtype=torch.float32)

    # Make prediction using the neural network model
    with torch.no_grad():
        predicted_output = model(input_tensor)

    # Extract the predicted value from the tensor and convert it to float
    predicted_output = predicted_output.item()

    # Update output label with the predicted output
    output_label.config(text=f'Speech Understanding: {predicted_output:.2f}')
# Create Tkinter application window
window = tk.Tk()
window.title('Speech Understanding')

# Load background image
# background_path = 'DT.png'  # Replace with your background image path
# background_img = Image.open('DT.png')
# background_img = background_img.resize((800, 600), background_img.ANTIALIAS)  # Resize image if needed
# background_photo = ImageTk.PhotoImage(background_img)

# Create widgets
background_label = tk.Label(window)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

title_label = tk.Label(window, text="Speech Understanding", font=('Helvetica', 24))
entry = tk.Entry(window, font=('Helvetica', 14))
predict_button = tk.Button(window, text="Predict", command=predict_output)
output_label = tk.Label(window, text="", font=('Helvetica', 14))

# Arrange widgets
title_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
entry.place(relx=0.5, rely=0.3, anchor=tk.CENTER, width=200)
predict_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
output_label.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

# Run Tkinter event loop
window.mainloop()
