import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score
# data_frame = pd.read_csv('EV_max_edit.csv', delimiter=';')
df = pd.read_csv('ruhe.csv', delimiter= ';' , encoding='iso-8859-1')

string_array = df.iloc[:, :-1].values.astype(str)


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
float_array = vectorized_convert(string_array)
df_float = pd.DataFrame(float_array, columns=df.columns[:-1])
print(df_float.shape)
correlation_matrix = df_float.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", xticklabels=False, yticklabels=correlation_matrix.columns, cbar_kws={'orientation': 'vertical'})
plt.yticks(rotation=45)
plt.title('Correlation Matrix')
plt.savefig('cr.png')
plt.show()

highly_correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            highly_correlated_features.add(colname)

print(highly_correlated_features)
df_filtered = df.drop(columns=highly_correlated_features)
print(df_filtered.shape)

# Save the filtered DataFrame to a new CSV file if needed
df_filtered.to_csv('filtered_file.csv', index=False)
label_encoder= LabelEncoder()
# data_frame['Ankopl_limks'] = laber_encoder.fit_transform(data_frame['Ankopplung_links'])
# print(data_frame[['Ankopplung_links', 'Ankopl_limks']])
# data_frame['HG_Erfahrung'] = data_frame['HG_Erfahrung'].replace('2019,5' , '2019')

print("Remaining features:")
print(df_filtered.columns.tolist())

# data_frame.to_csv('modified_3.csv' , index=False)
