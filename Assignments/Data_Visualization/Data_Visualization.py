import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('tips.csv')    # reading the dataset

print(data.head())    # printing the top 5 rows

# scatter plot 1 with day against trip
plt.scatter(data['day'], data['tip'], c=data['size'], s=data['total_bill'])
plt.title('Scatter Plot 1')    # add title for plot
plt.xlabel('Day')
plt.ylabel('Tip')
plt.colorbar()
plt.show()

# scatter plot 2
plt.plot(data['tip'])
plt.plot(data['size'])
plt.title('Scatter Plot 2')
plt.xlabel('Day')
plt.ylabel('Tip')
plt.show()
