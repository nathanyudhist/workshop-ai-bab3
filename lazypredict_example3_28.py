# Example 3.28: The LazyPredict.ipynb Program (Part A-E, digabung)

# --- Part A: Install library (jalankan sekali di terminal, bukan sebagai kode) ---
# pip install lazypredict

# --- Part B: LazyPredict Classification demo (dataset Iris) ---
import lazypredict
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1)

clf = LazyClassifier()
models_clf, predictions_clf = clf.fit(X_train, X_test, y_train, y_test)
print(models_clf)

# --- Part C: Plot hasil classification ---
plt.figure(figsize=(10, 5))
plt.plot(models_clf.index, models_clf['Accuracy'])
plt.xticks(rotation=90)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison (Iris)')
plt.tight_layout()
plt.show()

# --- Part D: LazyPredict Regression demo (dataset California housing) ---
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=1)

reg = LazyRegressor()
models_reg, predictions_reg = reg.fit(X_train, X_test, y_train, y_test)
print(models_reg)

# --- Part E: Plot hasil regression ---
plt.figure(figsize=(10, 5))
plt.plot(models_reg.index, models_reg['R-Squared'], '-s')
plt.xticks(rotation=90)
plt.ylabel('R-Squared')
plt.title('Model R-Squared Comparison (California Housing)')
plt.tight_layout()
plt.show()