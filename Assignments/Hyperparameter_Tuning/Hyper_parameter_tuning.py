import pandas as pd
from sklearn import svm, datasets
iris_dataset = datasets.load_iris()
df = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
df['flower'] = iris_dataset.target
df['flower'] = df['flower'].apply(lambda x: iris_dataset.target_names[x])
print(df)
print(df.columns)
