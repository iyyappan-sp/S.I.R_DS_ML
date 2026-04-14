import warnings
warnings.filterwarnings('ignore')
import joblib
fish_weight = joblib.load('weight_predict.pkl')
for i in range(2):
    print("Enter the value for fish weight predeiction")
    Vertical = int(input("Enter your vertical value:"))
    Diagonal = int(input("Enter your diagonal value:"))
    Cross = int(input("Enter your cross value:"))
    Height = int(input("Enter your height value:"))
    Width = int(input("Enter your width value:"))

    prediction = fish_weight.predict([[Vertical, Diagonal, Cross, Height, Width]])
    print(prediction)