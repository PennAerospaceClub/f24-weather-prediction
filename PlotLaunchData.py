import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_csv('PastLaunchData.csv')
columns_to_drop = ['Unnamed: 1', 'MILLIS', 'ATM DIFF', 'altitude (meters)', 'TEMP.1', 'EULERX', 'EULERY', 'EULERZ', 'latitude (degrees)', 'longitude (degrees)', 'ALT', 'COURSE', 'SPEED', 'NUM SATS', 'VEL DIFF']
df = df.drop (columns = columns_to_drop, axis = 1)
df[['TEMP', 'HUMIDITY', 'PRESSURE']].plot(title='TEMP HUMIDITY AND PRESSURE VS TIME', figsize=(10,10), color = ['red', 'blue', 'yellow'])
plt.show()
