# -*- coding: utf-8 -*-
"""PAC Weather Prediction

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fLzJgUMWTj6y6gqluc-0-VzuMufthjHB

This notebook aims to create a Random Forest model that will predict temperature given latitude, longitude, altitude, humidity, pressure.

**Add *PastLaunchData.csv* to the notebook files**

# Cleaning the Data

We will first clean and scale the data. Here, we will:
*   drop any unnecessary columns
  * including duplicate columns and weather balloon flight data
*   seperate our data into features and target variables
*   fill in any missing data with the median
*   split data for test/train
*   scale the data
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

# Load the data
file_path = "data\PastLaunchData.csv"
data = pd.read_csv(file_path)

# Drop unnecessary columns (like unnamed or duplicated temperature columns)
data = data.drop(columns=["Unnamed: 1", "TEMP.1", "EULERX", "EULERY", "EULERZ",
                          "COURSE", "NUM SATS", "VEL DIFF"], errors='ignore')

# Convert all columns to numeric, coercing errors to NaN
data = data.apply(pd.to_numeric, errors='coerce')

# Fill missing values with the median of each column
data.fillna(data.median(), inplace=True)

# Separate features and target variable
X = data.drop(columns=["TEMP"], errors='ignore')  # Features (exclude TEMP)
y = data["TEMP"]  # Target variable (temperature)

# Convert TIME column to numeric, handle any non-numeric entries by coercing them to NaN
X['TIME'] = pd.to_numeric(X['TIME'], errors='coerce')

# Fill NaN values in the TIME column with the median value of the column
X['TIME'].fillna(X['TIME'].median(), inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""# Training the Random Forest Model
In this section, we will use our cleaned data to train a Random Forest model.
"""

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)