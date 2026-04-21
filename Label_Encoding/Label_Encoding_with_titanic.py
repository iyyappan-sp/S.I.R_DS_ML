import pandas as pd
data = pd.read_csv('titanic_dataset.csv')
print(data.info())
print(data)
print(data.describe())
print(data.isnull().sum())
data['Age'] = data['Age'].fillna(data['Age'].mean())
print(data.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le_gender = LabelEncoder()
data['gender'] = le_gender.fit_transform(data['Gender'])    # use LabelEncoder
# use mapping concepts
#data['gender'] = data['gender'].map({'female': 0, 'male': 1})
print(le_gender.classes_)
data = data.drop('Gender', axis = 1)

X = data[['Pclass', 'gender', 'Age', 'Fare']]
y = data['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state =42)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print(y_predict)

y_predict_df = pd.DataFrame(y_predict, columns = ['Predicted_Survived'])
print(y_predict_df)

y_test_df = pd.DataFrame(y_test)
print(y_test_df)    # it will show random values
y_test_df = y_test_df.reset_index(drop = True)
print(y_test_df)

result_df = pd.concat([y_test_df, y_predict_df], axis = 1)
print(result_df)    # our model make wrong predictions

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_predict)
print(confusion)

import matplotlib.pyplot as plt
import seaborn as sns
data = {'y_Actual': y_test, 'y_Predicted': y_predict}
df = pd.DataFrame(data, columns = ['y_Actual', 'y_Predicted'])
clf_confusion_mat = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames = ['Actual'], colnames = ['Predicted'])
sns.heatmap(clf_confusion_mat, annot = True , fmt='d')
plt.show()

dt_model_score = clf.score(X_test, y_test)
print('Decision Tree Model Score',dt_model_score)

from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(n_estimators = 400)
clf2.fit(X_train, y_train)
y_pred_rf = clf2.predict(X_test)
print(y_pred_rf)

confusion_mat_rf = confusion_matrix(y_test, y_pred_rf)
print(confusion_mat_rf)

data = {'y_Actual': y_test, 'y_Predicted': y_pred_rf}
df = pd.DataFrame(data, columns = ['y_Actual', 'y_Predicted'])
clf2_confusion_mat_rf  = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames = ['Actual'], colnames = ['Predicted'])
sns.heatmap(clf2_confusion_mat_rf, annot = True , fmt = 'd')
plt.show()

rf_model_score = clf2.score(X_test, y_test)
print('Random Forest Model Score',rf_model_score)
