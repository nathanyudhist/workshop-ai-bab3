from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=1),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = model.score(X_test, y_test)   # R^2 score

best = max(results, key=results.get)
for name, score in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{name}: {score:.4f}")
print("\nBest model:", best)