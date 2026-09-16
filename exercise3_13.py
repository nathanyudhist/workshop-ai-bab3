# Exercise 3.13
# Modify Example 3.20 (K-means) to use sklearn.datasets.make_blobs() to
# generate the sample data instead of a manual array.
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=2, n_features=3, random_state=0)

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_[:10], "...")
print(kmeans.cluster_centers_)
print(kmeans.predict([[0, 0, 0]]))
