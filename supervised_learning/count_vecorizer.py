from sklearn.feature_extraction.text import CountVectorizer
#20 newsgroups -- Usenet / netnews

from sklearn.datasets import fetch_20newsgroups
ng = fetch_20newsgroups()
#(1) from the ng bunch create X and y
X = ng.data
y = Series(ng.target)

# (2) Create a vocabulary vector from our X
cv = CountVectorizer()
cv.fit(X)

# Create a matrix from our X as well
X_matrix = cv.transform(X)
# create a model
model = LogisticRegression()
#results = cross_val_score(model, X_matrix, y)
#results.mean()

X_train, X_test, y_train, y_test = train_test_split(X_matrix, y)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_pred, y_test)

# train the model with X, y

# predict with the model
documents = ['''The deal proposed by Senate leaders would lock in a two-year agreement on spending numbers, essentially taking shutdowns threats and continuing resolutions off the table for the time being. It would provide long overdue disaster relief funding to hurricane-ravaged Texas, Florida and, most urgently, Puerto Rico. Congress has been lurching from spending crisis to spending crisis for seemingly months. Should the House and Senate pass this, it in large part clears the decks.
But there are still major steps that need to be taken, so expect an exciting next 24 hours.''']

document_matrix = cv.transform(documents)
predict_name = model.predict(document_matrix)

# display type 
ng.target_names[predict_name]

# Another prediction

documents = ['''The deal proposed by Senate leaders would lock in a two-year agreement on spending numbers, essentially taking shutdowns threats and continuing resolutions off the table for the time being. It would provide long overdue disaster relief funding to hurricane-ravaged Texas, Florida and, most urgently, Puerto Rico. Congress has been lurching from spending crisis to spending crisis for seemingly months. Should the House and Senate pass this, it in large part clears the decks.
But there are still major steps that need to be taken, so expect an exciting next 24 hours.''']

document_matrix = cv.transform(documents)
model.predict(document_matrix)

