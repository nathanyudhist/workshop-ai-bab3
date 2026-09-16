# Exercise 3.9
# Modify Example 3.12 to perform random forest "classification" on the
# diabetes dataset.
#
# Note: diabetes.target is a continuous disease-progression score (25-346),
# not a class label, so a classifier treats each distinct value as its own
# class and accuracy will look poor. This follows the exercise as written;
# for a meaningful model here, RandomForestRegressor + R^2 would be the
# right tool instead.
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Total points: %d  Correctly labeled points: %d" %
      (y_test.shape[0], (y_test == y_pred).sum()))
