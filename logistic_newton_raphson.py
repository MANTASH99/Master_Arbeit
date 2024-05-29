
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

data_frame = pd.read_csv('EV_max_edit.csv', delimiter=';')

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
#output_data11 = vectorized_convert(data_frame.iloc[:, -1].values)
# Print the array with float elements


input_data1 = float_array
output_data1 = data_frame.iloc[:, -1].values
input_data2 = torch.from_numpy(input_data1).float()
output_data2 = torch.from_numpy(output_data1).view(-1, 1).float()
input_train, input_test, output_train, output_test = train_test_split(input_data2, output_data2, test_size=0.2)

class LogisticRegressionNewton:
    def __init__(self, max_iter=100, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        X = np.c_[np.ones((m, 1)), X]  # Add a bias term (intercept)

        self.theta = np.zeros((n + 1, 1))

        for i in range(self.max_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (y - h))
            hessian = -np.dot(X.T, X * h * (1 - h))

            delta_theta = np.linalg.solve(hessian, gradient)
            self.theta += delta_theta

            if np.linalg.norm(delta_theta) < self.tol:
                break

    def predict(self, X):
        m, n = X.shape
        X = np.c_[np.ones((m, 1)), X]  # Add a bias term (intercept)
        z = np.dot(X, self.theta)
        return np.round(self.sigmoid(z))

# Example usage
if __name__ == '__main__':
    # Generate some synthetic data
    np.random.seed(0)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # Create and train the logistic regression model
    model = LogisticRegressionNewton()
    model.fit(input_train, output_train)

    # Predict using the trained model
    y_pred = model.predict(input_test)

    # Calculate accuracy
    accuracy = np.mean(y_pred == output_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")