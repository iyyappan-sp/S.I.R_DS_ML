import warnings
warnings.filterwarnings("ignore")

import joblib
salary = joblib.load('salary_model.pkl')
for i in range(3):

    print('Enter the values for salary prediction')
    experience = int(input("Enter your experience:"))
    test_score = int(input("Enter your test score:"))
    interview_score = int(input("Enter your interview score:"))

    # predict using the model
    predict_val = salary.predict([[experience, test_score, interview_score]])
    print("Predicted salary:",predict_val[0])