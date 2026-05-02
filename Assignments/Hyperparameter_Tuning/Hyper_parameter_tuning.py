import pandas as pd
from sklearn import svm, datasets
iris_dataset = datasets.load_iris()
df = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
df['flower'] = iris_dataset.target
df['flower'] = df['flower'].apply(lambda x: iris_dataset.target_names[x])
print(df)
print(df.columns)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, test_size = 0.3)

model = svm.SVC(kernel='rbf', C=30, gamma='auto')
model.fit(X_train, y_train)
model.score(X_test, y_test)

from sklearn.model_selection import cross_val_score
avg_score_linear = cross_val_score(svm.SVC(kernel='linear', C=10, gamma='auto'), iris_dataset.data, iris_dataset.target, cv=5)
print(avg_score_linear)

