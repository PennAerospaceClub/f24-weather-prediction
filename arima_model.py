# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_csv('PastLaunchData.csv')
start_time = pd.Timestamp("2022-01-01 00:00:00")

df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
df['TIME'] = pd.to_timedelta(df['TIME'], unit='s')
df['TIME'] = start_time + df['TIME']

df = df.dropna(subset=['TIME', 'TEMP', 'PRESSURE', 'altitude (meters)'])
df.set_index('TIME', inplace=True)

fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
df['TEMP'].plot(ax=axes[0], title='Temperature')
df['HUMIDITY'].plot(ax=axes[1], title='Humidity')
df['PRESSURE'].plot(ax=axes[2], title='Pressure')
df['altitude (meters)'].plot(ax=axes[3], title='Altitude (meters)')
plt.xlabel('Time')
plt.show()

# TODO: Split the data into training, development, and test data using a 1:3:10 ratio. Think about what dimension each of these should be
total_num_rows = len(df)

train_data = int(total_num_rows * (10/14))
test_data = int(total_num_rows * (4/14))

train = df[:train_data]
test = df[train_data:]

# Defining the target and exogenous variables for training and testing
train_target = train['TEMP']
train_exog = train[['PRESSURE', 'altitude (meters)']]
test_exog = test[['PRESSURE', 'altitude (meters)']]

# Initialize ARIMA model with exogenous predictors
model = ARIMA(train_target, exog=train_exog, order=(5,1,0))  # Order (p,d,q) may need tuning
model_fit = model.fit()
#print(model_fit.summary())


# # line plot of residuals
# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot(title='Residuals')
# plt.show()
# # density plot of residuals
# residuals.plot(kind='kde', title='Residual Density')
# plt.show()
# # summary stats of residuals
# print(residuals.describe())

test_predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, exog=test_exog)

test_predictions.index = test.index
additional_steps = 3463

# Generate future predictions
future_predictions = model_fit.forecast(steps=additional_steps, exog=test_exog[:additional_steps])

# Create a datetime index for future predictions
future_index = pd.date_range(test.index[-1] + pd.Timedelta(seconds=1), periods=additional_steps, freq='s')
future_predictions.index = future_index

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train.index, train_target, label='Training Data', color='blue')
plt.plot(test.index, test['TEMP'], label='Actual Temperature (Test Data)', color='green')
plt.plot(test.index, test_predictions, label='Predicted Temperature (Test Range)', color='red')
plt.plot(future_predictions.index, future_predictions, label='Future Predictions', color='purple')

# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature Prediction with Extended Forecast')
plt.legend()
plt.show()
