import pandas as pd
import numpy as np
import re
import random
import sklearn
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib as plt
a = np.zeros((800, 12))
data_frame = pd.read_csv('pokemon_data.csv')

# print(data_frame) give you al the data


# print(data_frame.values)  you an array of the data as an matrix which is good
b = data_frame.values
# print(data_frame.head(8)) gives as the first 8 rows with all columns i can also add values at the end so i
# have it as a matrix
# you can do the opposite for head , by calling tail , it will start from bellow
# col = data_frame.columns
# col_data = data_frame['Name']
# col_list= col_data.to_list()
# print(col_list)
# print(col) i ca n have the columns
# print(col[]) a certain value from the column
# print(data_frame['Name'][0:5]) #you will have the first 5 rows from the column NAME ,
# print(data_frame[['Name' , 'Type 1']][0:5]) you got the point right

# print(data_frame.iloc[2 , 1 ]) #so you can locate every point inside the set that ou need
# print(data_frame)
# for index, row in data_frame.iter rows(): iterate over all rows side by side , you can also
# specify which row by print(index , row['name])
# data_frame.loc[data_frame['type 1'] == "gras" ] gives you all raws hat have those conditions
# print(data_frame.describe()) stats uber den
#
# print(data_frame)
# print(data_frame.sort_values('Name', ascending=True)) , sorting by a col , if asc  =  false its wil start from bellow
# print(data_frame.sort_values(['Speed' , 'Generation'  , 'HP'], ascending=[1,0,1]))
# create a new col :
data_frame['T'] = data_frame['HP'] + data_frame['Speed']
# print(data_frame) you will get a new row in the data set
data_frame = data_frame.drop(columns=['T'])  # dropping columns
data_frame['T'] = data_frame.iloc[:, 4:10].sum(axis=1)
# print(data_frame)
# data_frame = data_frame.drop(columns=['T'])
# Columns_of_data_set = list(data_frame.columns.values)
# print(Columns_of_data_set) down sorting columns
# data_frame = data_frame[Columns_of_data_set[0:4] + [Columns_of_data_set[-1]] + Columns_of_data_set[4:12]]
# print(data_frame)
data_frame.to_csv('modified_panbdas.csv', index=False)  # save my modified data
#  done playing around with data , now to more advanced stuff
# note i can use any data_frame to save as a csv and then take a lok there more easily
# you can reset the index of the modified csv after going back to csv by new_df = new_df.reset_index(drop = true)
#                    FIlITING THE DATA , THERE IS MORE ON  THE VEDIO THAT I WE WATCHED
contains = data_frame.loc[data_frame['Name'].str.contains('Mega')]
not_contains = data_frame.loc[~data_frame['Name'].str.contains('Mega')]
contains_ = data_frame.loc[data_frame['Type 1'].str.contains('fire|grass', flags=re.I, regex=True)]
contains = contains.reset_index(drop=True)
#print(contains.shape, contains)
#                                       CONDITIONAL CHANGES
#  data_frame.loc[data_frame['Type 1']=='Flamer' , 'Type 1'] = 'Fire'   # change some names goes like that
#                                       Group by aggregate Statistics



