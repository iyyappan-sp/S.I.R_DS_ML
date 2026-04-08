import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('students_performance.csv')
print(df)
print(df.isnull().sum())

# --- EDA ---
df.gender.hist()
plt.show()

df.math_score.hist()
plt.show()

df.reading_score.hist()
plt.show()

# Boxplot for math_score
sns.boxplot(df['math_score'])  # before reducing outliers
plt.show()

df = df[df.math_score >= 30]   # after reducing outliers
sns.boxplot(df['math_score'])
plt.show()

# Boxplot for reading_score
sns.boxplot(df['reading_score'])  # before reducing outliers
plt.show()

df = df[df.reading_score >= 30]   # after reducing outliers
sns.boxplot(df['reading_score'])
plt.show()

# Keep relevant columns
df = df[['gender','test_preparation_course','math_score','reading_score','writing_score']]
print(df)
print(df.shape)
print(df.isna().sum())

# Add new column 'total_score'
df['total_score'] = df['math_score'] + df['reading_score'] + df['writing_score']
print(df)

# Add new column 'avg_score' (corrected calculation)
df['avg_score'] = df['total_score'] / 3
print(df)

# --- Compare average scores by test preparation course ---
avg_scores = df.groupby('test_preparation_course')['avg_score'].mean().reset_index()
print(avg_scores)

# Bar plot
sns.barplot(x='test_preparation_course', y='avg_score', data=avg_scores)
plt.title("Average Score by Test Preparation Course")
plt.ylabel("Average Score")
plt.xlabel("Test Preparation Course")
plt.show()
