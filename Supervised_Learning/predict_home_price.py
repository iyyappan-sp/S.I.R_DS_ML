import warnings
warnings.filterwarnings('ignore')

import joblib
home_price = joblib.load('home_price_model.pkl')
"""
predict_val = home_price.predict([[3000,3,40]])
print(predict_val)

predict_val = home_price.predict([[2500,4,5]])
print(predict_val)
"""

# try it for input from the user
#looping concepts
for i in range(2):
    print('Enter the values for home price prediction')
    area = int(input("Enter your area size sq.ft:"))
    bedrooms = int(input("Enter your bedrooms:"))
    age = int(input("Enter your age:"))

    # predict using the model
    predict_val = home_price.predict([[area, bedrooms, age]])
    print("Predicted home price:",predict_val[0])