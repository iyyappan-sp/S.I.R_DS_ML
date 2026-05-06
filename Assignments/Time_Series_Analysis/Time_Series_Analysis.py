import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv("AirPassengers.csv")
print(data)

# Convert to datetime
data['Month'] = pd.to_datetime(data['Month'])
indexedData = data.set_index('Month')
print(indexedData)

# Plot original data
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.plot(indexedData)
plt.show()

# Rolling statistics
rolmean = indexedData.rolling(window=12).mean()
rolstd = indexedData.rolling(window=12).std()
print(rolmean, rolstd)

plt.plot(indexedData, color='red', label='Original')
plt.plot(rolmean, color='green', label='Rolling Mean')
plt.plot(rolstd, color='blue', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean and Rolling Std')
plt.show()

# Dickey-Fuller Test
from statsmodels.tsa.stattools import adfuller
print("Results of Dickey-Fuller Test:")
dftest = adfuller(indexedData['#Passengers'])

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' % key] = value

print(dfoutput)

# Log transformation
indexedData_logScale = np.log(indexedData)
plt.plot(indexedData_logScale)
plt.show()

# Moving average
movingAvg = indexedData_logScale.rolling(window=12).mean()
movingSTD = indexedData_logScale.rolling(window=12).std()

plt.plot(indexedData_logScale)
plt.plot(movingAvg, color='red')
plt.show()
