# Exercise 3.19
# Modify Example 3.27 (PyCaret classification demo) to use the breast cancer
# dataset instead of Iris.
#
# NOTE: PyCaret only supports Python 3.9-3.11 and its dependencies (pandas
# 2.1.4) fail to compile on Python 3.14. Run this in a separate virtual
# environment created with Python 3.9-3.11 (see the setup steps discussed
# earlier in this chat).
from sklearn import datasets
from pycaret import classification

cancer = datasets.load_breast_cancer(as_frame=True)
cancer.data['Target'] = cancer.target
df = cancer.data

classification.setup(data=df, target='Target')
classification.compare_models()
