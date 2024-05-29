import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

data_frame = pd.read_csv('bi_all_1 - Kopie.csv' , delimiter=';' , encoding='iso-8859-1')
strin_array = data_frame.iloc[1: , 0].values.astype(str)
string_output = data_frame.iloc[1: , -2].values.astype(str)


def convert_to_float(value):
    try:
        return float(value.replace(',', '.'))
    except ValueError:
        return np.nan


# Vectorize the conversion function
vectorized_convert = np.vectorize(convert_to_float)

HV = vectorized_convert(strin_array)
output = vectorized_convert(string_output)
HV_train , HV_test , pred_train , pred_test = train_test_split(HV , output , test_size=0.2)
beta0 = 5.99
beta1 = -0.0793
def exp(HV , beta0, beta1):
    return  100*((np.exp(beta0+beta1*HV)))/(1+np.exp(beta0+beta1*HV))

#def deltai() absolute value of xi - x prediction , then find the median wich is not the same as the mean



