# ensemble models 

# Gradient Boost Regressor
# Random Forest Regressor


# (1) Load the boston data
# (2) Create an X data frame (from inputs) and a y series (from targets)
# (3) use train_test_split to create training, testing X/y
# (4) Use LinearRegression to create and train a model
# (5) Use mean_squared_error to check your model's predictive power


from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.model_selection import LeaveOneOut

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import load_boston
boston = load_boston()

X = DataFrame(boston.data, columns=boston.feature_names)
y = Series(boston.target)

for estimator in [LinearRegression, KNeighborsRegressor,
                 DecisionTreeRegressor,
                 GradientBoostingRegressor,
                 RandomForestRegressor]:
    model = estimator()
    results = cross_val_score(model, X, y, 
                          cv=LeaveOneOut(),
                          scoring='neg_mean_squared_error')
    print("{0}: {1}".format(estimator.__name__,
                            results.mean()))

model = GradientBoostingRegressor()
model.fit(X, y)


one_house = X.loc[10]
one_house

model.predict([one_house])


model = GradientBoostingRegressor()
model.fit(X, y)
model.score(X, y)
