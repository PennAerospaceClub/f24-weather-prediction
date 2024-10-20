# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv('PastLaunchData.csv')

df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce')
df = df.dropna(subset=['TIME', 'TEMP', 'HUMIDITY', 'PRESSURE'])
df.set_index('TIME', inplace=True)

df[['TEMP', 'HUMIDITY', 'PRESSURE']].plot(title='Temp, Humidity, Pressure', figsize=(10,10))
plt.show()

# TODO: Split the data into training, development, and test data using a 1:3:10 ratio. Think about what dimension each of these should be
train_data

total_num_rows = len(df)

