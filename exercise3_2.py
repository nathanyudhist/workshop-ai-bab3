# Exercise 3.2
# Modify Example 3.2 to use the 3rd and 4th features (petal length, petal
# width) instead of the first two (sepal length, sepal width).
from sklearn import svm, datasets

iris = datasets.load_iris()
X = iris.data[:, 2:4]      # petal length, petal width
y = iris.target            # 0: Setosa, 1: Versicolour, 2: Virginica

clf = svm.SVC()
clf.fit(X, y)

p = clf.predict([[1.4, 0.2]])
print("Prediction using petal length/width:", p)

# Comparison note:
# Example 3.2 used sepal length/width. Sepal measurements overlap more
# between Versicolour and Virginica, so an SVM trained on sepal features
# tends to be less accurate than one trained on petal features, which
# separate the three Iris species much more cleanly.
