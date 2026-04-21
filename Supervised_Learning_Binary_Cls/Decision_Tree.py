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
print(result_df)    # model make wrong prediction

from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print(confusion_mat)

import matplotlib.pyplot as plt
import seaborn as sns
data  = {'y_Actual': y_test, 'y_Predicted': y_pred}
df = pd.DataFrame(data, columns = ['y_Actual', 'y_Predicted'])
# Actual first(Rows), Predicted second(Columns)
clf_confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(clf_confusion_matrix, annot = True)
plt.show()

from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(n_estimators=200)    # how many trees are in my forest
clf2.fit(X_train, y_train)
y_pred_rf = clf2.predict(X_test)
print(y_pred_rf)

confusion_mat_rf = confusion_matrix(y_test, y_pred_rf)
print(confusion_mat_rf)

data  = {'y_Actual': y_test, 'y_Predicted': y_pred_rf}
df = pd.DataFrame(data, columns = ['y_Actual', 'y_Predicted'])
# Actual first(Rows), Predicted second(Columns)
clf_confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(clf_confusion_matrix, annot = True)
plt.show()

import joblib
joblib.dump(clf2, 'diabetes_rf.pkl')