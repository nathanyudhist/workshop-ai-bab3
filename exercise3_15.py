# Exercise 3.15
# (Same note as Exercise 3.14.) Add a third group of points, with only one
# labeled point per group.
import numpy as np
from sklearn.semi_supervised import LabelSpreading

X = np.array([
    [0, 1], [1, 1], [2, 0], [3, 1],          # group 0
    [10, 5], [11, 6], [12, 4], [13, 5],      # group 1
    [20, 15], [21, 16], [22, 14], [23, 15],  # group 2 (new, well separated)
])

labels = np.full(12, -1.)
labels[0] = 0   # one labeled point in group 0
labels[4] = 1   # one labeled point in group 1
labels[8] = 2   # one labeled point in group 2
print("Labels before:", labels)

# n_neighbors is lowered to 3 (fewer than points per group) so the
# label spreading stays local to each group instead of bleeding across.
label_spread = LabelSpreading(kernel='knn', n_neighbors=3, alpha=0.8)
label_spread.fit(X, labels)
print("Labels after:", label_spread.transduction_)
