import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline

data = pd.read_csv("StudentsPerformance.csv")

print(data.head())
data.describe()

data.info()
data.columns = ['Gender','Race_eth','Parent_education','Lunch',
                'Test_prep','Math_score','Reading_score','Writing_score']

plt.figure(figsize=(10,7))
sns.countplot(data['Gender'])

plt.figure(figsize=(10,7))
sns.countplot(data["Race_eth"])

plt.figure(figsize=(10,7))
sns.countplot(data["Parent_education"])

sns.countplot(data["Test_prep"])

plt.figure(figsize=(10,7))
sns.pairplot(data)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = data.drop("Writing_score", axis = 1)
X = pd.get_dummies(X)
y = data["Writing_score"]

x_train, x_test, y_train , y_test = train_test_split(X, y, test_size = 0.3)

lm1 = LinearRegression()
lm1_fit = lm1.fit(x_train, y_train)
lm1_predict = lm1.predict(x_test)

rmse = mean_squared_error(y_test, lm1_predict)
print(rmse)

fig, ax = plt.subplots()
ax.scatter(y_test, lm1_predict)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

enet = ElasticNet()
enet_fit = enet.fit(x_train, y_train)
enet_predict = enet.predict(x_test)

rmse1 = mean_squared_error(y_test, enet_predict)
print(rmse1)

fig, ax = plt.subplots()
ax.scatter(y_test, enet_predict)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

rf = RandomForestRegressor()
rf_fit = rf.fit(x_train, y_train)
rf_predict = rf.predict(x_test)

rmse2 = mean_squared_error(y_test, enet_predict)
print(rmse2)

fig, ax = plt.subplots()
ax.scatter(y_test, rf_predict)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()