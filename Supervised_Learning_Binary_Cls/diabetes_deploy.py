import joblib
model = joblib.load('diabetes_rf.pkl')

import pandas as pd
newdata = pd.read_csv('newdataset.csv')
print(newdata.info())
print(newdata)

predict = model.predict(newdata)
print(predict)
newdata['Outcome'] = predict
print(newdata)
newdata.loc[newdata['Outcome'] == 1, 'Outcome'] = 'Positive'
print(newdata)