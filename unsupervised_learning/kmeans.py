# Kmeans unsupervised learning
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture


iris = load_iris()

X = DataFrame(iris.data, columns=iris.feature_names)
y = Series(iris.target)


model = GaussianMixture(n_components=3, covariance_type='full')
model.fit(X)

y_pred = model.predict(X)


model = KMeans(n_clusters =3)
model.fit(X)
y_pred = model.predict(X)
Series(y_pred).value_counts()

# Find random score
from sklearn.metrics import adjusted_rand_score

adjusted_rand_score(y, y_pred)

#homogeneity score
from sklearn.metrics import homogeneity_score
homogeneity_score(y, y_pred)


#Completeness score
from sklearn.metrics import completeness_score
completeness_score(y, y_pred)


