# read version from installed package
from importlib.metadata import version
__version__ = version("regressors_grano")

from .regressor_ensemble import EnsembleRegressor, class_dict
from .regressor_grad_boost import GradBoostRegressor
from .regressor_lasso import LassoRegressor
from .regressor_nearest_neighbor import KNeigRegressor
from .regressor_ordinary_least_squares import OrdinaryLeastSquaresRegressor
from .regressor_ridge_regression import RidgeRegressionRegressor

__all__ = [
    # Classes you want to be accessible when <from yourpackage import *> is used.
    'EnsembleRegressor',
    'class_dict',
    'GradBoostRegressor',
    'LassoRegressor',
    'KNeigRegressor',
    'OrdinaryLeastSquaresRegressor',
    'RidgeRegressionRegressor'
]