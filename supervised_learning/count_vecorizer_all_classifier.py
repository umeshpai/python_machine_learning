from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

input_text = ['Buy a sandwitch',
             'Buy lunch because I like lunch',
             'I would like to eat lunch',
             'I am buying lunch for lunch eaters',
             'Machine learning which sklearn is fun',
             'I like data science and machine learning',
             'Python is the best ever for data science',
             'I like data science',
             'Data science my favarable',
             'Lunch to be eaten']

Categories = ['lunch', 'data science']

y=[0,0,0,0,1,1,1,1,1,0]

cv = CountVectorizer()
cv.fit(input_text)

X = cv.transform(input_text)

for estimator in [KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier,
                 RandomForestClassifier]:
    model = estimator()
    result = cross_val_score(model, X, y)
    print("{0}: {1}".format(estimator.__name__, results.mean()))
    


model.fit(X,y)

# PREDICT now
evaluate_text_matrix = cv.transform(['Data science is fun',
                                        'I am hungry. Let\'s eat!',
                                      'Lunch at data science for machine learning'])
model.predict(evaluate_text_matrix)

#evaluate again
evaluate_text_matrix = cv.transform(['adhajsd akdh adkh adh',
                                        'I am hungry. Let\'s eat!',
                                      'Lunch at data science for machine learning'])
model.predict(evaluate_text_matrix)

