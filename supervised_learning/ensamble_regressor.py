#Ensemble modeles using X

from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import scale

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

for estimator in [LinearRegression, KNeighborsRegressor, DecisionTreeRegressor,
                 RandomForestRegressor, GradientBoostingRegressor]:
    model = estimator()
    results = cross_val_score(model, X, y, cv=LeaveOneOut(),
                             scoring='neg_mean_squared_error')
    print("{0}: {1}".format(estimator.__name__, results.mean()))


