import warnings
warnings.filterwarnings("ignore")
import joblib
fish_width = joblib.load('width_prediction.pkl')

for i in range(3):
    print("Enter the values for fish width prediction")
    weight = int(input("Enter the weight value :"))
    vertical = int(input("Enter the vertical value :"))
    diagonal = int(input("Enter the diagonal value :"))
    cross = int(input("Enter the cross value :"))
    height = int(input("Enter the height value :"))

    prediction = fish_width.predict([[weight, vertical, diagonal, cross, height]])
    print(prediction)