# Exercise 3.14
# NOTE ON THE BOOK TEXT: the exercise says "modify the program from
# Example 3.20", but adding points to *labeled groups* only makes sense for
# the semi-supervised Example 3.21 (label spreading) right above it in the
# same section -- Example 3.20 is plain K-means and has no notion of
# "labeled" points. This looks like an error in the book, so the solution
# below extends Example 3.21 instead: add two more points to each of the
# two existing groups.
import numpy as np
from sklearn.semi_supervised import LabelSpreading

X = np.array([
    [0, 1], [1, 1], [2, 0], [3, 1], [1, 0], [2, 2],        # group 0 (6 points)
    [10, 5], [11, 6], [12, 4], [13, 5], [11, 4], [12, 6],  # group 1 (6 points)
])

labels = np.full(12, -1.)
labels[0] = 0     # only the first point of group 0 is labeled
labels[-1] = 1    # only the last point of group 1 is labeled
print("Labels before:", labels)

label_spread = LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(X, labels)
print("Labels after:", label_spread.transduction_)
