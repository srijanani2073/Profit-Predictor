import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('50_Startups.csv')

X = data[['R&D Spend', 'Administration', 'Marketing Spend']]
X.columns = ['R&D Spend', 'Administration', 'Marketing Spend']
y = data['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)

all_data_pred = model.predict(X)

for i in range(len(data)):
    print("Actual Profit:", data['Profit'][i], "Predicted Profit:", all_data_pred[i])

accuracy = model.score(X, y)
print("Accuracy:", accuracy)

rd_spend = float(input("Enter R&D Spend: "))
admin_cost = float(input("Enter Administration Cost: "))
marketing_spend = float(input("Enter Marketing Spend: "))

user_input = [[rd_spend, admin_cost, marketing_spend]]
user_pred = model.predict(user_input)

print(f"Predicted Profit for User Input: {user_pred[0]:.2f}")
