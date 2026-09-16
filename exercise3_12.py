# Exercise 3.12
# Modify Example 3.18 (multiple linear regression) to use Scikit-Learn's
# Linnerud dataset. X = [chins, situps, jumps]; y has 3 targets
# [weight, waist, pulse] -- we regress on the first target (weight).
from sklearn import linear_model
from sklearn.datasets import load_linnerud

linnerud = load_linnerud()
X = linnerud.data
y = linnerud.target[:, 0]     # weight

reg = linear_model.LinearRegression()
reg.fit(X, y)
print('Coefficients:', reg.coef_)
print('Intercept:', reg.intercept_)
print('Prediction for first sample:', reg.predict([X[0]]))
