import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('tips.csv')    # reading the dataset

print(data.head())    # printing the top 5 rows

# scatter plot 1 with day against trip---matplotlib
plt.scatter(data['day'], data['tip'], c=data['size'], s=data['total_bill'])
plt.title('Scatter Plot 1 using matplotlib')    # add title for plot
plt.xlabel('Day')
plt.ylabel('Tip')
plt.colorbar()
plt.show()

# scatter plot 2---matplotlib
plt.plot(data['tip'])
plt.plot(data['size'])
plt.title('Scatter Plot 2 using matplotlib')
plt.xlabel('Day')
plt.ylabel('Tip')
plt.show()

# bar chart with day against trip---matplotlib
plt.bar(data['day'], data['tip'])
plt.title("Bar Chart using matplotlib")
plt.xlabel('Day')
plt.ylabel('Tip')
plt.show()

# histogram of total bills-matplotlib
plt.hist(data['total_bill'])
plt.title('Histogram using matplotlib')
plt.show()

# draw lineplot---seaborn
sns.lineplot(x="sex", y="total_bill", data=data)
plt.title('Lineplot 1 using seaborn')    # sns---seaborn , plt---matplotlib
plt.show()

# scatter plot using seaborn
sns.scatterplot(x='day', y='tip', data=data, hue='sex')
plt.title('Scatter Plot using seaborn')
plt.show()

# lineplot using seaborn
sns.lineplot(x='day', y='tip', data=data)
plt.title("Lineplot 2 using seaborn")
plt.show()

# seaborn using only data attribute
sns.lineplot(data=data.drop(['total_bill'], axis=1))
plt.title('Lineplot 2 using seaborn')
plt.show()

# bar plot using seaborn
sns.barplot(x='day', y='tip', data=data, hue='sex')
plt.show()

# histogram plot using seaborn
sns.histplot(x='total_bill', data=data, kde=True, hue='sex')
plt.title('Histogram plot using seaborn')
plt.show()

""" We See A New Library That Is Called Bokeh (pip install bokeh) """

from bokeh.plotting import figure, output_file, show
from bokeh.palettes import magma
import logging
from bokeh.util.warnings import BokehUserWarning
import warnings
warnings.filterwarnings("ignore", category=BokehUserWarning)


# instantiating the figure object
graph = figure(title = "Bokeh Scatter Graph")
color = magma(256)
# scatter plotting the graph using bokeh
graph.scatter(data['total_bill'], data['tip'], color=color)
show(graph)    # displaying the model on browser

# instantiating the figure object
graph = figure(title = "Bokeh Bar Chart")
# count of each unique values of tip column
df = data['tip'].value_counts()
# plotting the graph
graph.line(df, data['tip'])    # lineplot
show(graph)    # displaying the model on browser

# bar chart are two types: hbar() and vbar()
# hbar()---horizontal bar & vbar()---vertical bar
graph.vbar(data['total_bill'], top=data['tip'])    # plotting the graph
show(graph)    # displaying the model on browser

# Interactive Data Visualization
# Interactive Legends---click_policy property makes the legend interactive
graph.vbar(data['total_bill'], top=data['tip'], legend_label = "Bill VS Tips", color='green')
graph.vbar(data['tip'], top=data['size'], legend_label = "Tips VS Size", color='red')
graph.legend.click_policy = "hide"
show(graph)    # displaying the model on browser


""" We See A New Library That Is Called Plotly (pip install plotly) """

import plotly.express as px

# plotting the scatter chart using plotly
fig = px.scatter(data, x="day", y="tip", color='sex')
fig.show()    # showing the plot on browser

# plotting the line plot using plotly
fig = px.line(data, y='tip', color='sex')
fig.show()    # showing the plot on browser

# plotting the bar chart using plotly
fig = px.bar(data, x='day', y='tip', color='sex')
fig.show()    # showing the plot on browser

# plotting the histogram plot using plotly
fig = px.histogram(data, x='total_bill', color='sex')
fig.show()    # showing the plot on browser

# Adding interaction on plotly--->Creating Dropdown Menu
import plotly.graph_objects as px
plot = px.Figure(data=[px.Scatter(x=data['day'],y=data['tip'],mode='markers',)])
# Add dropdown
plot.update_layout(
    updatemenus=[
        dict(
            buttons=list([
                dict(
                    args=["type", "scatter"],
                    label="Scatter Plot",
                    method="restyle"
                ),
                dict(
                    args=["type", "bar"],
                    label="Bar Chart",
                    method="restyle"
                )
            ]),
            direction="down",
        ),
    ]
)

plot.show()