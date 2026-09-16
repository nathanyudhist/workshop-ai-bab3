# Exercise 3.7
# Modify Example 3.10 to perform PCA on the breast cancer data instead of Iris.
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('mean radius')
plt.ylabel('mean texture')
plt.title('Original Breast Cancer Data')

pca = decomposition.PCA(n_components=3)
pca.fit(X)
X1 = pca.transform(X)

plt.figure(2)
plt.scatter(X1[:, 0], X1[:, 1], c=y)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA-Transformed Breast Cancer Data')
plt.show()
