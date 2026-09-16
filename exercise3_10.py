# Exercise 3.10
# Modify Example 3.14 (comparison of classifiers) to use the diabetes dataset.
# Same caveat as Exercise 3.9 applies; QDA can fail outright since a "class"
# may have just one sample, so it's wrapped in try/except.
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

names = ["SVM", "Naive Bayes", "LDA", "QDA", "Decision Tree",
         "Random Forest", "Nearest Neighbors", "Neural Networks"]

classifiers = [
    SVC(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    MLPClassifier(alpha=1, max_iter=2000)]

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

for name, clf in zip(names, classifiers):
    try:
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(name + ": " + str(score))
    except ValueError as e:
        print(name + ": failed ->", e)
