import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
import seaborn as sns
# Custom conversion function
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
        # out = self.fc2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        # # # # # # out = self.drop_out(out)
        # # # # #
        # # # # #
        # # # # # # # out = self.drop_out(out)
        # # # # # #
        # out = self.fc2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        # # # # #
        # out = self.fc2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        # # # #
        # # # #
        # # # #
        # out = self.fc2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        # # # # # out = self.drop_out(out)
        # # # # # #
        # out = self.fc2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        # # # # # out = self.drop_out(out)
        # # # # # #
        # out = self.fc2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        # # #out = self.drop_out(out)
        # # out = self.fc2(out)
        # # out = self.bn2(out)
        # # out = self.relu(out)
        #
        # out = self.fc2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        #
        #
        #
        #
        # # out = self.drop_out(out)
        #
        #
        #


        out = self.fc3(out)
        # out = self.cliped_tensor(out)# use it at the end , not trainig
        return out
def register_hooks(module):
    def forward_hook(module, input, output):
        module.input_cache = input

    def backward_hook(module, grad_input, grad_output):
        input_data = module.input_cache[0]
        print("Gradient Shape:", grad_output[0].shape)
        print("Input Shape:", input_data.shape)

    hook_handles = []

    # Register forward hook
    hook_handles.append(module.register_forward_hook(forward_hook))

    # Register backward hook
    hook_handles.append(module.register_full_backward_hook(backward_hook))

    return hook_handles


model = NeuralNetwork(input_train.shape[1], 300 , 0.1)
hook_handles = register_hooks(model)
model.load_state_dict(torch.load(r"/Users/mahdimantash/Downloads/pathes/hearingloss_shap_experemnt.pth"))
model.eval()

#
# def GRADIENT_BASED():
#     data_point = input_test[0]  # Replace with the desired data point
#
#     # Calculate gradients with respect to input features
#     data_point = data_point.unsqueeze(0).requires_grad_()
#     outputs = model(data_point)
#     outputs.backward()
#     gradients = data_point.grad[0].numpy()
#     feature_importance = np.abs(gradients)
#     feature_names = ["4fpta", "gender", "125", "250", "500", "750", "1000", "2000", "3000", "4000", "6000", "7000",
#                      "8000"]
#     # Plot the gradient heatmap
#     plt.bar(feature_names, feature_importance)
#     plt.xticks(rotation='vertical')
#     plt.ylabel('Gradient Magnitude')
#     plt.title('Feature Importance')
#     plt.tight_layout()
#     plt.show()

#GRADIENT_BASED()
# to report , different data points are giving me different feuture importancy ,
# i would like latly not to forget , and maybe avarage over the data points , and see by summing or voting which had the more in fluency




def SHAP(model, input_train, input_test):
    explainer = shap.DeepExplainer(model, input_train)
    shap_values = explainer.shap_values(input_test)
    # Print SHAP values for the first instance in the test set

    print("SHAP Values:", shap_values[0])

    # You can also visualize the explanations using summary plots
    shap.summary_plot(shap_values, input_test, feature_names=data_frame.columns[:-1], show=False)

    plt.savefig('shap.png')
    plt.show()

# Call the SHAP function
SHAP(model, input_train, input_test)

# Remove the hooks after use
for handle in hook_handles:
    handle.remove()


####data correlation
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Heatmap')
# plt.show()