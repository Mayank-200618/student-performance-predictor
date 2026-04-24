# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'Attendance': [50, 55, 60, 65, 70, 75, 80, 90],
    'Previous_Score': [40, 45, 50, 55, 60, 65, 70, 80],
    'Final_Score': [42, 48, 52, 58, 63, 67, 72, 85]
}

df = pd.DataFrame(data)

# Features & Target
X = df[['Hours_Studied', 'Attendance', 'Previous_Score']]
y = df['Final_Score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predicted = model.predict(X_test)

print("Predictions:", predicted)

# Custom prediction
hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance: "))
prev_score = float(input("Enter previous score: "))

result = model.predict([[hours, attendance, prev_score]])
print("Predicted Final Score:", result[0])
