# Salary Prediction using Linear Regression

This project predicts employee salaries based on their experience, test scores, and interview scores using a Linear Regression model.

## 🚀 Features
* **Data Preprocessing:** Handles missing values and converts word-based numbers to integers.
* **Model Persistence:** Saves the trained model using `joblib` for real-time predictions.
* **Interactive CLI:** A separate script to input data and get instant salary predictions.

## 🛠️ Tech Stack
* **Python**
* **Pandas** (Data Manipulation)
* **Scikit-Learn** (Machine Learning)
* **Word2Number** (Text-to-Numeric conversion)

## 📂 Project Structure
* `LinearRegression_ML.py`: Script to clean data and train the model.
* `predict_salary.py`: Script to load the saved model and make predictions.
* `hiring_salaries.csv`: The dataset.
* `salary_model.pkl`: The serialized (saved) model.

## 📋 How to Run
1. Install dependencies: `pip install pandas scikit-learn word2number joblib`
2. Run training: `python LinearRegression_ML.py`
3. Run prediction: `python predict_salary.py`
