import warnings
warnings.filterwarnings('ignore')

import joblib
home_price = joblib.load('home_price_model.pkl')

predict_val = home_price.predict([[3000,3,40]])
print(predict_val)

predict_val = home_price.predict([[2500,4,5]])
print(predict_val)