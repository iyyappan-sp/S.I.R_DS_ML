import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import joblib
model = joblib.load('diabetes_rf.pkl')

print('Enter the values of diagnostic measurements')
Pregnancies = float(input("Enter the patient's Pregnancies values: "))
Glucose = float(input("Enter the patient's Glucose values: "))
BloodPressure = float(input("Enter the patient's BloodPressure values: "))
SkinThickness = float(input("Enter the patient's SkinThickness values: "))
Insulin = float(input("Enter the patient's Insulin values: "))
BMI = float(input("Enter the patient's BMI values: "))
DiabetesPedigreeFunction = float(input("Enter the patient's DiabetesPedigreeFunction values: "))
Age = int(input("Enter the patient's Age values: "))

input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
predict_val = model.predict(input_data)

if predict_val[0] == 1:
    result = 'Positive'
else:
    result = 'Negative'
print(f"\nPrediction Result: {result}")

"""
newdata = pd.read_csv('newdataset.csv')
print(newdata.info())
print(newdata)


predict = model.predict(newdata)
print(predict)
newdata['Outcome'] = predict
print(newdata)
newdata.loc[newdata['Outcome'] == 1, 'Outcome'] = 'Positive'
print(newdata)
"""