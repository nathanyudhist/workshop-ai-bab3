# Exercise 3.6
# Modify Example 3.8 to use 6 features and 2,000 samples. Compare with
# Example 3.8 (4 features, 1,000 samples).
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X, y = make_classification(n_samples=2000, n_features=6,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

print(clf.predict([[0, 0, 0, 0, 0, 0]]))
print("Training accuracy:", clf.score(X, y))

# Comparison note:
# More samples (2,000 vs 1,000) gives LDA a more stable estimate of each
# class's mean/variance. More features (6 vs 4) doesn't automatically
# help, since n_informative is still 2 -- the extra features are noise.
