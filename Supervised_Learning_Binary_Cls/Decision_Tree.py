import pandas as pd

dataset = pd.read_csv("diabetes.csv")
print(dataset)
print(dataset.info())
print(dataset.describe())
counts = dataset['Outcome'].value_counts()
print(counts)

X = dataset.iloc[:, 0:7]
y = dataset.iloc[:,8]
print(X)
print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=42)
diabetes_counts = y_train.value_counts()
print(diabetes_counts)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(y_pred)    # this will show upon numpy array
y_predict_df = pd.DataFrame(y_pred, columns = ['Predicted_value'])
print(y_predict_df)
y_test_df = pd.DataFrame(y_test)    # it is show allways random data's
print(y_test_df)
y_test_df = y_test_df.reset_index(drop = True)
print(y_test_df)
result_df = pd.concat([y_test_df, y_predict_df], axis =1)
print(result_df)