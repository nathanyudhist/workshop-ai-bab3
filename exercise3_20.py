# Exercise 3.20
# Modify the LazyPredict classification example (Example 3.28b) to use the
# wine dataset instead of Iris.
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1)

clf = LazyClassifier()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)
