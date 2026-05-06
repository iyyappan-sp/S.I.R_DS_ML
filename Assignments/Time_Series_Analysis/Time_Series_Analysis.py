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

# Remove trend
datasetLogScaleMinusMovingAvg = indexedData_logScale - movingAvg
print(datasetLogScaleMinusMovingAvg.head(12))
datasetLogScaleMinusMovingAvg.dropna(inplace=True)
print(datasetLogScaleMinusMovingAvg.head(10))

# Stationarity test function
def test_stationarity(timeseries):
    movingAvg = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()

    plt.plot(timeseries, color='red', label='Original')
    plt.plot(movingAvg, color='green', label='Rolling Mean')
    plt.plot(movingSTD, color='blue', label='Rolling Std')

    plt.legend(loc='best')
    plt.title('Rolling Mean and Rolling Std')
    plt.show()

    print("Results of Dickey-Fuller Test:")
    result = adfuller(timeseries)

    output = pd.Series(result[0:4],
                       index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key, value in result[4].items():
        output['Critical Value (%s)' % key] = value

    print(output)

# Apply stationarity test
test_stationarity(datasetLogScaleMinusMovingAvg)
# Exponential weighted moving average
expAvg = indexedData_logScale.ewm(halflife=12).mean()

plt.plot(indexedData_logScale)
plt.plot(expAvg, color='red')
plt.show()

datasetLogScaleMinusExpAvg = indexedData_logScale - expAvg
test_stationarity(datasetLogScaleMinusExpAvg)

# Differencing
datasetLogDiffshifting = indexedData_logScale - indexedData_logScale.shift()
plt.plot(datasetLogDiffshifting)
plt.show()

datasetLogDiffshifting.dropna(inplace=True)
test_stationarity(datasetLogDiffshifting)

# Decomposition (Trend, Seasonal, Residual)
from statsmodels.tsa.seasonal import seasonal_decompose
decomposed = seasonal_decompose(indexedData_logScale, period=12)

trend = decomposed.trend
seasonal = decomposed.seasonal
residual = decomposed.resid

plt.subplot(411)
plt.plot(indexedData_logScale, label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label='Residual')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

# ARIMA MODEL
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(indexedData_logScale, order=(2,1,2))
results_ARIMA = model.fit()
print(results_ARIMA.summary())

# Plot fitted values
plt.plot(indexedData_logScale, label='Log Data')
plt.plot(results_ARIMA.fittedvalues, color='red', label='Fitted')
plt.legend(loc='best')
plt.title('ARIMA Fit')
plt.show()

# Forecast
forecast = results_ARIMA.forecast(steps=12)
# Convert back to original scale
forecast_values = np.exp(forecast)

print("Forecast Values")
print(forecast_values)

# Plot forecast
plt.plot(indexedData, label='Original')
plt.plot(forecast_values, color='red', label='Forecast')
plt.legend(loc='best')
plt.title('Final Forecast')
plt.show()