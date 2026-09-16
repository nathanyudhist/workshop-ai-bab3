# Exercise 3.18
# Modify Example 3.26 (auto_ml regression on the Boston dataset) to use the
# California housing dataset instead.
#
# NOTE: the 'auto_ml' library used in Example 3.26 has been unmaintained
# since ~2018 and fails to install on modern Python (confirmed: it needs
# pandas<1.0, which cannot be built on Python 3.12+). If you just need a
# working "automatic ML" tool today, use the LazyPredict version below
# instead (same idea: try many models automatically, then rank them).

from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

reg = LazyRegressor()
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)

# --- Original book code (needs auto_ml, likely won't install on modern Python) ---
# from auto_ml import Predictor
# df = X.copy()
# df['MedHouseVal'] = y
# df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
# column_descriptions = {'MedHouseVal': 'output'}
# ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)
# ml_predictor.train(df_train)
# ml_predictor.score(df_test, df_test.MedHouseVal)
