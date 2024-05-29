import pandas as pd
import numpy as np
data_frame = pd.read_csv('you.csv', delimiter=';', encoding='iso-8859-1')

# Get the third row as new column names

new_column_names = data_frame.iloc[1].astype(str).tolist()
print(new_column_names[39])
# Skip the first two rows (header and second row)
new_data_frame = data_frame.copy()
for i in range(39):
    new_data_frame.rename(columns={new_data_frame.columns[i]: new_column_names[i]}, inplace=True)
for i in range(39,54):
    new_data_frame.rename(columns={new_data_frame.columns[i]: new_column_names[i] + 'NAL50dB/TSPL'}, inplace=True)
for j in range(55,70):
    new_data_frame.rename(columns={new_data_frame.columns[j]: new_column_names[j] + 'NAL65dB/TSPL'}, inplace=True)
for i in range(71,86):
    new_data_frame.rename(columns={new_data_frame.columns[i]: new_column_names[i] + 'NAL80dB/TSPL'}, inplace=True)
for j in range(86,106):
    new_data_frame.rename(columns={new_data_frame.columns[j]: new_column_names[j] + 'HGOWN50dB(REAR SPL )'}, inplace=True)
for i in range(107,126):
    new_data_frame.rename(columns={new_data_frame.columns[i]: new_column_names[i] + 'HGOWN65dB(REAR SPL )'}, inplace=True)
for j in range(127,146):
    new_data_frame.rename(columns={new_data_frame.columns[j]: new_column_names[j] + 'HGOWN80dB(REAR SPL )'}, inplace=True)
for i in range(147,166):
    new_data_frame.rename(columns={new_data_frame.columns[i]: new_column_names[i] + 'HGOWN50dB(99.PerzentilSPL)'}, inplace=True)
for j in range(167,186):
    new_data_frame.rename(columns={new_data_frame.columns[j]: new_column_names[j] + 'HGOWN65dB(99.PerzentilSPL)'}, inplace=True)
for i in range(187,206):
    new_data_frame.rename(columns={new_data_frame.columns[i]: new_column_names[i] + 'HGOWN80dB(99.PerzentilSPL)'}, inplace=True)
for j in range(207,226):
    new_data_frame.rename(columns={new_data_frame.columns[j]: new_column_names[j] + 'HGOWN50dB(30.PerzentilSPL)'}, inplace=True)
for j in range(227,246):
    new_data_frame.rename(columns={new_data_frame.columns[j]: new_column_names[j] + 'HGOWN65dB(30.PerzentilSPL)'}, inplace=True)
for i in range(247,266):
    new_data_frame.rename(columns={new_data_frame.columns[i]: new_column_names[i] + 'HGOWN80dB(30.PerzentilSPL)'}, inplace=True)
for j in range(267,286):
    new_data_frame.rename(columns={new_data_frame.columns[j]: new_column_names[j] + 'HGOWN50dB(Verstärkung)'}, inplace=True)
for i in range(287,306):
    new_data_frame.rename(columns={new_data_frame.columns[i]: new_column_names[i] + 'HG OWN65dB(Verstärkung)'}, inplace=True)
for j in range(307,326):
    new_data_frame.rename(columns={new_data_frame.columns[j]: new_column_names[j] + 'HGOWN80dB(Verstärkung)'}, inplace=True)
for j in range(327,346):
    new_data_frame.rename(columns={new_data_frame.columns[j]: new_column_names[j] + 'HGOWN50dB(Kompressionsrate)'}, inplace=True)
for i in range(347,366):
    new_data_frame.rename(columns={new_data_frame.columns[i]: new_column_names[i] + 'HGOWN65dB(Kompressionsrate)'}, inplace=True)
for j in range(367,386):
    new_data_frame.rename(columns={new_data_frame.columns[j]: new_column_names[j] + 'HGOWN80dB(Kompressionsrate)'}, inplace=True)

new_data_frame.to_csv('named_bi_all.csv' , index=False )




