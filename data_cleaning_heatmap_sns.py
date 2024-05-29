import torch
import torch.nn as nn
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score
# data_frame = pd.read_csv('EV_max_edit.csv', delimiter=';')
df = pd.read_csv('Book2.csv', delimiter=';',  encoding='iso-8859-1')
string_array = df.iloc[:, :].values.astype(str)
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
df1 = pd.DataFrame(float_array , columns=df.columns)
# df1['5REUG'] = df1[['10000dB SPL ' , '1000dB SPL' , '4000dB SPL ' , '1600dB SPL ' , '5000dB SPL ']].mean(axis=1)
df1['5NAL_50'] = df1[['630NAL50dB/TSPL' , '400NAL50dB/TSPL' , '500NAL50dB/TSPL' , '1000NAL50dB/TSPL' , '1600NAL50dB/TSPL']].mean(axis=1)
df1['5NAL_65'] = df1[['5000HGOWN65dB(REAR SPL )' , '500NAL65dB/TSPL' , '630NAL65dB/TSPL' , '800NAL65dB/TSPL' , '1000NAL65dB/TSPL']].mean(axis=1)
df1['5NAL_80'] = df1[['500NAL80dB/TSPL' , '400NAL80dB/TSPL' , '630NAL80dB/TSPL' , '800NAL80dB/TSPL' , '250NAL80dB/TSPL']].mean(axis=1)
df1['5HG_OWN_50'] = df1[['2000HGOWN50dB(REAR SPL )' , '4000HGOWN50dB(REAR SPL )' , '6300HGOWN50dB(REAR SPL )' , '5000HGOWN50dB(REAR SPL )' , '3150HGOWN50dB(REAR SPL )']].mean(axis=1)
df1['5HG_OWN_65'] = df1[['5000HGOWN65dB(REAR SPL )' , '8000HGOWN65dB(REAR SPL )' , '6300HGOWN65dB(REAR SPL )' , '1000HGOWN65dB(REAR SPL )' , '3150HGOWN65dB(REAR SPL )']].mean(axis=1)
df1['5HG_OWN_80'] = df1[['1250HGOWN80dB(REAR SPL )' , '5000HGOWN80dB(REAR SPL )' , '8000HGOWN80dB(REAR SPL )' , '10000HGOWN80dB(REAR SPL )' , '4000HGOWN80dB(REAR SPL )']].mean(axis=1)
df1['5HG_OWN_99P_50'] = df1[['125HGOWN50dB(99.PerzentilSPL)' , '5000HGOWN50dB(99.PerzentilSPL)' , '5000HGOWN50dB(99.PerzentilSPL)' , '10000HGOWN50dB(99.PerzentilSPL)' , '8000HGOWN50dB(99.PerzentilSPL)']].mean(axis=1)
df1['5HG_OWN_99P_65'] = df1[['125HGOWN65dB(99.PerzentilSPL)' , '5000HGOWN65dB(99.PerzentilSPL)' , '8000HGOWN65dB(99.PerzentilSPL)' , '1250HGOWN65dB(99.PerzentilSPL)' , '10000HGOWN65dB(99.PerzentilSPL)']].mean(axis=1)
df1['5HG_OWN_99P_80'] = df1[['400HGOWN80dB(99.PerzentilSPL)' , '800HGOWN80dB(99.PerzentilSPL)' , '10000HGOWN80dB(99.PerzentilSPL)' , '8000HGOWN80dB(99.PerzentilSPL)' , '5000HGOWN80dB(99.PerzentilSPL)']].mean(axis=1)
df1['5HG_OWN_30P_50'] = df1[['6300HGOWN50dB(30.PerzentilSPL)' , '4000HGOWN50dB(30.PerzentilSPL)' , '5000HGOWN50dB(30.PerzentilSPL)' , '3150HGOWN50dB(30.PerzentilSPL)' , '1250HGOWN50dB(30.PerzentilSPL)']].mean(axis=1)
df1['5HG_OWN_30P_65'] = df1[['800HGOWN65dB(30.PerzentilSPL)' , '4000HGOWN65dB(30.PerzentilSPL)' , '6300HGOWN65dB(30.PerzentilSPL)' , '8000HGOWN65dB(30.PerzentilSPL)' , '5000HGOWN65dB(30.PerzentilSPL)']].mean(axis=1)
df1['5HG_OWN_30P_80'] = df1[['3150HGOWN80dB(30.PerzentilSPL)' , '8000HGOWN80dB(30.PerzentilSPL)' , '5000HGOWN80dB(30.PerzentilSPL)' , '6300HGOWN80dB(30.PerzentilSPL)' ]].mean(axis=1)
df1['5HG_OWN_verstäkung_50'] = df1[['400HGOWN50dB(V' , '4000HGOWN50dB(V' , '8000HGOWN50dB(V' , '6300HGOWN50dB(V' , '1250HGOWN50dB(V']].mean(axis=1)
df1['5HG_OWN_verstäkung_65'] = df1[['125HG OWN65dB(V' , '630HG OWN65dB(V' , '10000HG OWN65dB(V' , '6300HG OWN65dB(V' , '1600HG OWN65dB(V']].mean(axis=1)
df1['5HG_OWN_verstäkung_80'] = df1[['500HGOWN80dB(V' , '3150HGOWN80dB(V' , '10000HGOWN80dB(V' , '8000HGOWN80dB(V' ]].mean(axis=1)
df1['5HG_OWN_Kompressionsrate_50'] = df1[['125HGOWN50dB(Kompressionsrate)' , '315HGOWN50dB(Kompressionsrate)' , '250HGOWN50dB(Kompressionsrate)' , '8000HGOWN50dB(Kompressionsrate)' , 'Unnamed: 346']].mean(axis=1)
df1['5HG_OWN_Kompressionsrate_65'] = df1[['125HGOWN65dB(Kompressionsrate)' , '315HGOWN65dB(Kompressionsrate)' , '400HGOWN65dB(Kompressionsrate)' , '2500HGOWN65dB(Kompressionsrate)' , '10000HGOWN65dB(Kompressionsrate)']].mean(axis=1)
df1['5HG_OWN_Kompressionsrate_80'] = df1[['125HGOWN80dB(Kompressionsrate)' , '315HGOWN80dB(Kompressionsrate)' , '400HGOWN80dB(Kompressionsrate)' , '5000HGOWN80dB(Kompressionsrate)' , '500HGOWN80dB(Kompressionsrate)']].mean(axis=1)




df1.to_csv('average5.csv', sep=';', index=False, float_format='%.3f')



# Custom conversion function
# def convert_to_float(value):
#     try:
#         # If it's already a float, return it as is
#         if isinstance(value, float):
#             return value
#         # Otherwise, convert the string to float
#         return float(value.replace(',', '.'))
#     except ValueError:
#         return np.nan
#
#
# # Vectorize the conversion function
# vectorized_convert = np.vectorize(convert_to_float)
# float_array = vectorized_convert(string_array)
# df_float = pd.DataFrame(float_array, columns=df.columns[:-1])
# print(df_float.shape)
# correlation_matrix = df_float.corr()
# highly_correlated_features = set()
# for i in range(len(correlation_matrix.columns)):
#     for j in range(i):
#         if abs(correlation_matrix.iloc[i, j]) > 0.8:
#             colname = correlation_matrix.columns[i]
#             highly_correlated_features.add(colname)

# Drop the highly correlated features
# df_filtered = df.drop(columns=highly_correlated_features)
# print(df_filtered.shape)
#
# # Save the filtered DataFrame to a new CSV file if needed
# df_filtered.to_csv('filtered_file.csv', index=False)
# label_encoder= LabelEncoder()
# data_frame['Ankopl_limks'] = laber_encoder.fit_transform(data_frame['Ankopplung_links'])
# print(data_frame[['Ankopplung_links', 'Ankopl_limks']])
# data_frame['HG_Erfahrung'] = data_frame['HG_Erfahrung'].replace('2019,5' , '2019')
#
#
# data_frame.to_csv('modified_3.csv' , index=False)




# print((data_frame_point))
# data_frame_point.to_csv('dor_out_bi_all_11.csv', index=False, float_format='%.2f')
# with open('3fpta.csv', 'w', newline='', encoding='iso-8859-1') as csvfile:
#     csv_writer = csv.writer(csvfile, delimiter=';')
#
#     # Write the header
#     csv_writer.writerow(df1.columns)
#
#     # Write the data
#     for row in df1.values:
#         csv_writer.writerow(row)



# output_data1 = data_frame.iloc[:, -1].values
# input_data2 = torch.from_numpy(input_data1).view(-1,1).float()
# output_data2 = torch.from_numpy(output_data1).view(-1, 1).float()
# input_train, input_test, output_train, output_test = train_test_split(input_data2, output_data2, test_size=0.2)
# # input_val, input_test, output_val, output_test = train_test_split(input_train, output_train, test_size=0.2)
# plt.figure(figsize=(10, 8))
# sns.heatmap(vs, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Heatmap')
# plt.savefig('HG OWN 80 dB (Kompressionsrate).png')
# plt.show()


