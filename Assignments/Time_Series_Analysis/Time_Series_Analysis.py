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
