# -------- Import Libraries --------
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------- Data Preprocessing --------
df=pd.read_csv("50_Startups.csv")
df.head()
df.isnull().sum()
df.describe()
df.info()
df=df.drop_duplicates()

# -------- Train-Test Split and Model Training --------
x=df[["R&D Spend","Marketing Spend"]]
y=df["Profit"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
# -------- Create Regression Plane --------
m = np.linspace(x['R&D Spend'].min(), x['R&D Spend'].max(), 10)
n = np.linspace(x['Marketing Spend'].min(), x['Marketing Spend'].max(), 10)

mG, nG = np.meshgrid(m, n)

final = np.vstack((mG.ravel(), nG.ravel())).T
z = lr.predict(final).reshape(10,10)
# -----------Display Metrixes-----------
print("Mean Absolute Error:",mean_absolute_error(y_test,y_pred))
print("Mean Squared Error:",mean_squared_error(y_test,y_pred))
print("R2 Score:",r2_score(y_test,y_pred))
# -------- Plot 3D Scatter --------
fig = px.scatter_3d(
    df,
    x='R&D Spend',
    y='Marketing Spend',
    z='Profit'
)
# -------- Add Regression Plane --------
fig.add_trace(go.Surface(
    x=mG,
    y=nG,
    z=z,
    opacity=0.5
))
fig.update_layout(
    title="Multiple Linear Regression: Startup Profit Prediction",
)
fig.show()
