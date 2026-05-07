import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# load dataset
data = pd.read_csv("diabetes.csv")

print(data.head())
print(data.columns)
