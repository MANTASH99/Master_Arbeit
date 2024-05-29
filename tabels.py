import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
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

# print(data_frame)
# Vectorize the conversion function
vectorized_convert = np.vectorize(convert_to_float)
float_array = vectorized_convert(string_array)
import pandas as pd

# Sample data
models = ['Model 1', 'Model 2', 'Model 3']
losses = [0.1, 0.15, 0.2]  # Example loss values for each model
r2_scores = [0.75, 0.80, 0.85]  # Example R2 scores for each model

# Create DataFrame
data = {'Model': models, 'Loss': losses, 'R2 Score': r2_scores}
df = pd.DataFrame(data)

# Display DataFrame
print(df)


# Display DataFrame with styling
display(df.style.set_caption('Model Comparison').format(precision=2))