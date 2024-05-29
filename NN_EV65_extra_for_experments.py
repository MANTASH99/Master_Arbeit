
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

data_f = pd.read_csv('ahh.csv' , delimiter=';', encoding='iso-8859-1')
string_array = data_f.iloc[:, :].values.astype(str)


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
vectorized_convert = np.vectorize(convert_to_float)
float_array = vectorized_convert(string_array)

values_array =data_f.values
columns  =  data_f.columns

frequencies = ['250', '315', '400', '500', '630', '800', '1000', '1250', '1600',
       '2000', '2500', '3150', '4000', '5000', '6300', '8000']
len_fre  = len(frequencies)
values_in_db = float_array[20]
print(values_in_db)
float_frequencies = [float(frequency) for frequency in frequencies]
fig, ax = plt.subplots(figsize=(20, 10))
bars = ax.bar(frequencies, values_in_db, color='black', width=0.1,label='db values at each frequency' )
ax.set_title('NAL 50 DB' , fontsize=26)
ax.set_xlabel('Frequency in  HZ' ,color='black',fontsize=23)
ax.set_ylabel('dB_SPL' , fontsize=23)
ax.set_xticks(range(len_fre) , labels = frequencies)
ax.set_xticklabels(labels= frequencies,fontsize=26)
tops = [bar.get_height() for bar in bars]
print(tops)

ax.plot(frequencies, tops, marker='o', color='red' , linestyle='dashed',linewidth=2 , label=None )

ax.set_xticklabels(frequencies, rotation='vertical', fontsize=26 , color='black' ,)
ax.set_yticklabels(ax.get_yticks(), fontsize=20)
fre_low = 500.0
fre_high = 4000.0
y_limit_low = min(values_in_db)
y_limit_high = 0

frequencies_float = [float(freq) for freq in frequencies]
print(tops)
tops1=[43.4, 42.8, 42.5]
ax.fill_between(['400', '500', '630'] , tops1 , y_limit_high , where=None , color='gold' , label='area of important frequencies')

tops2=[37.8, 39.9, 50.7]
ax.fill_between(['1000', '1250', '1600'] , tops2 , y_limit_high , where=None , color='gold' )
# ax.annotate('Important Area', xy=('500', 20), xytext=('550', 45),
#             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=14)

# ax.annotate('Important Area 2', xy=('1250', 20), xytext=('1400', 50),
#             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=25)

ax.legend(fontsize=26)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('NAL 50_1.png')
plt.show()


