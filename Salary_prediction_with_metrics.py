#---------------Importing all neccesary libraries---------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
#---------Data Preprocessing------------------------------
df=pd.read_csv("Salary_dataset.csv")
df.head()
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
#------------------Train - Test the data-------------------
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
# ----------------------Ploting Graph----------------------
x_line = x_train.iloc[:,0]
y_line = lr.predict(x_train)

sorted_index = x_line.argsort()

plt.scatter(x_line, y_train, color="red")
plt.plot(x_line.iloc[sorted_index], y_line[sorted_index], color="green")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience")
plt.show()
#-------Displaying MAE,MSE,RMSE,R-Square values----------
print("Mean Absolute Error Value is : ",mean_absolute_error(y_test,y_pred)) 
print("Mean Square Error Value is : ",mean_squared_error(y_test,y_pred)) 
print("Root Mean Absolute Error Value is : ",np.sqrt(mean_squared_error(y_test,y_pred)))
print("R-Square Value is : ",r2_score(y_test,y_pred))
