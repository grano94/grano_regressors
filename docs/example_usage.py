from regressors_grano import OrdinaryLeastSquaresRegressor
from sklearn.datasets import load_diabetes
from regressors_grano import EnsembleRegressor

# prepare data
data = load_diabetes() 
X = data.data
y = data.target

# evaluate single regressor
ls = OrdinaryLeastSquaresRegressor()
ls.fit(X, y)
ls.score(X, y)

# evaluate ensemble regressor
ensemble_regr = EnsembleRegressor(["gb", "rr"])
ensemble_regr.fit(X, y)
score = ensemble_regr.score(X, y)