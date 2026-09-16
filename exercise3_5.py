# Exercise 3.5
# Modify Example 3.7 so that it trains the Naive Bayes model, saves it to a
# file, loads it back, and makes a prediction with the loaded model.
import joblib
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)

clf = GaussianNB()
clf.fit(X, y)

joblib.dump(clf, 'naive_bayes_model.pkl')      # serialization

clf2 = joblib.load('naive_bayes_model.pkl')    # deserialization
p = clf2.predict([[5.0, 3.4, 1.5, 0.4]])
print(p)
