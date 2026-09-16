# Exercise 3.4
# Modify Example 3.6 so that it plots the histogram of radius, size (area),
# texture, and smoothness of all the data points.
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer(as_frame=True)
df = cancer.frame

features = ['mean radius', 'mean area', 'mean texture', 'mean smoothness']
df[features].hist(figsize=(8, 6))
plt.tight_layout()
plt.show()
