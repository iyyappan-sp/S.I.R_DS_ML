import joblib
salary = joblib.load('salary_model.pkl')

predict_val = salary.predict([[2,9,6]])    # experience,test_score_10,interview_score_10
print(predict_val)

predict_val = salary.predict([[12,10,10]])
print(predict_val)