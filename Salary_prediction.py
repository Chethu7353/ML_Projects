import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Salary_dataset.csv")

x = df[['YearsExperience']]
y = df['Salary']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

plt.scatter(df["YearsExperience"], df["Salary"])
plt.xlabel("Years")
plt.ylabel("Salary")

lr = LinearRegression()
lr.fit(x_train, y_train)

year = float(input("Enter the No year Experience : "))

p = lr.predict(pd.DataFrame([[year]], columns=['YearsExperience']))

m = lr.coef_
c = lr.intercept_

print("The slope of the regression line:", m)
print("The intercept of the graph:", c)
print("The expected Salary is:", p[0])
