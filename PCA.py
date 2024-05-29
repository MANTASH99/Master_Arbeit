import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
data_frame = pd.read_csv('bi_all.csv', delimiter=';' , encoding='iso-8859-1')

string_array = data_frame.iloc[:, :].values.astype(str)


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
# output_data11 = vectorized_convert(data_frame.iloc[:, -1].values)
# Print the array with float elements


input_data1 = float_array[3:,91:100]
input_df = pd.DataFrame(input_data1, columns=data_frame.columns[91:100])








scaler = StandardScaler()
X_scaled = scaler.fit_transform(input_data1)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Extract feature names from the first row of your CSV
feature_names = list(data_frame.columns[91:100])  # Adjust the column range as needed

# Visualize the correlation between the first 15 features and the first 15 principal components
num_features_to_visualize = 15
subset_feature_names = feature_names[:num_features_to_visualize]

# Create DataFrames for easier visualization
df_original = pd.DataFrame(X_scaled[:, :num_features_to_visualize], columns=feature_names)

# Concatenate the DataFrames


# Calculate the correlation matrix
subset_correlation_matrix = df_original.corr()

# Visualize the subset of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(subset_correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation between Original Features and Principal Components (Subset)')
plt.savefig('NAL_50db_target.png')
plt.show()
explained_variance_ratio = pca.explained_variance_ratio_

# Visualize the explained variance ratio
# plt.subplot(2, 1, 2)  # Create a subplot for the explained variance ratio plot
# plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.75, align='center', label='Individual explained variance')
# plt.step(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), where='mid', label='Cumulative explained variance')
# plt.xlabel('Principal Components')
# plt.ylabel('Explained Variance Ratio')
# plt.title('Explained Variance Ratio by Principal Components')
# plt.xticks(range(1, len(explained_variance_ratio) + 1), feature_names[:len(explained_variance_ratio)])  # Use actual feature names
# plt.legend()
#
# plt.tight_layout()  # Adjust layout to prevent overlapping
# plt.show()