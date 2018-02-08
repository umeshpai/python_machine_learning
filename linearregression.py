from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

boston = load_boston()

X = DataFrame(boston.data, columns=boston.feature_names)
y = Series(boston.target)

X_train, X_test, y_train, y_test = train_test_split(X,y)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

df = DataFrame({'pred':y_pred,
               'test': y_test})
mean_squared_error(y_pred, y_test)
df.plot.scatter(x='pred', y='test')

