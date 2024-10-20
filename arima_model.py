# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_csv('PastLaunchData.csv')

df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce', format='%y-%m-%d %H:%M:%S')
df = df.dropna(subset=['TIME', 'TEMP', 'HUMIDITY', 'PRESSURE'])
df.set_index('TIME', inplace=True)

df[['TEMP', 'HUMIDITY', 'PRESSURE']].plot(title='Temp, Humidity, Pressure', figsize=(10,10))
plt.show()

# TODO: Split the data into training, development, and test data using a 1:3:10 ratio. Think about what dimension each of these should be
total_num_rows = len(df)

train_data = int(total_num_rows * (1/14))
dev_data = int(total_num_rows * (3/14))
test_data = int(total_num_rows * (10/14))

train = df[:train_data]
dev = df[train_data:train_data + dev_data]
test = df[train_data + dev_data:]

model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

print(model_fit.summary())