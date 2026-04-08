import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px

data = pd.read_csv("/content/cancer_classification.csv")

X = data.drop('benign_0__mal_1', axis=1)
y = data['benign_0__mal_1']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5
)

model.fit(X_train, y_train)

print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

importance = pd.Series(model.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False).head(5))

plt.figure(figsize=(20,10))
tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Benign', 'Malignant'],
    filled=True
)
plt.show()

fig = px.scatter_3d(
    data,
    x='mean concave points',
    y='worst perimeter',
    z='worst texture',
    color=data['benign_0__mal_1'].map({0: 'Benign', 1: 'Malignant'}),
    title='3D Visualization of Cancer Data'
)

fig.update_traces(marker=dict(size=4))
fig.show()
