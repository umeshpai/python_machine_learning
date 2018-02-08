# (1) Load burrito_current.csv into Pandas, and create X and y
# (2) Clean it up -- remove the "const" column, check for NaN
# (3) Check the scale, with a boxplot
# (4) Load a few regression models, and try them out using
# cross_val_score
# (5) Which model has the most predictive power?
# (6) Where can a burrito restaurant owner skimp, and still get
# a good score?


df = pd.read_csv('burrito_current.csv',
                usecols=[2,3,4,5,6,7,8,9,10,11,12])
X = df.drop('rating', axis=1)
y = df['rating']
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

for estimator in [LinearRegression, KNeighborsRegressor,
                 DecisionTreeRegressor, 
                 GradientBoostingRegressor,
                 RandomForestRegressor]:
    model = estimator()
    results = cross_val_score(model, X, y, cv=LeaveOneOut(),
                             scoring='neg_mean_squared_error')
    print("{0}: {1}".format(estimator.__name__,
                           results.mean()))



#Predict
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
burrito_predictions = DataFrame({'pred':y_pred,
                                'train':y})
burrito_predictions.plot.scatter(x='pred', y='train')

mean_squared_error(y_pred, y)

