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

avg_score_rbf_1 = cross_val_score(svm.SVC(kernel='rbf', C=10, gamma='auto'), iris_dataset.data, iris_dataset.target, cv=5)
print(avg_score_rbf_1)

avg_score_rbf_2 = cross_val_score(svm.SVC(kernel='rbf', C=20, gamma='auto'), iris_dataset.data, iris_dataset.target, cv=5)
print(avg_score_rbf_2)

import numpy as np
kernel = ['rbf', 'linear']
C = [1,10,20]
avg_scores = {}
for kval in kernel:
    for cval in C:
        cv_scores = cross_val_score(svm.SVC(kernel=kval, C=cval, gamma='auto'), iris_dataset.data, iris_dataset.target, cv=5)
        avg_scores[kval + '_' + str(cval)] = float(np.average(cv_scores))
print(avg_scores)

# Gridsearch
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(svm.SVC(gamma='auto'), {
    'C': [1,10,20],
    'kernel': ['rbf', 'linear']
}, cv=5, return_train_score=False)
clf.fit(iris_dataset.data, iris_dataset.target)
print(clf.cv_results_)

df = pd.DataFrame(clf.cv_results_)
print(df.head())
print(df[['param_C', 'param_kernel', 'mean_test_score']])

print(dir(clf))
print(clf.best_score_)
print(clf.best_params_)

# RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(svm.SVC(gamma='auto'), {
    'C': [1, 10, 20],
    'kernel': ['rbf', 'linear']
    },
    cv=5,
    return_train_score=False,
    n_iter=2
)
rs.fit(iris_dataset.data, iris_dataset.target)
rs_result_score = pd.DataFrame(rs.cv_results_)[['param_C', 'param_kernel', 'mean_test_score']]
print(rs_result_score)
