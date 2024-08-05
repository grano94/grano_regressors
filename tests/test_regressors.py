"""
File name: test_regressors.py
Author: Andreas-Nizar Granitzer
Date: July 2024
Description: 
    This file provides functionality for running tests on the regressor classes. 
    
"""

from regressors_grano import EnsembleRegressor, class_dict
from regressors_grano import RegressorBase
from regressors_grano import GradBoostRegressor
from regressors_grano import RidgeRegressionRegressor
import pytest
import numpy as np

## fixtures 
@pytest.fixture # decorator for reuse of objects
def ensemble_regressor():
    return EnsembleRegressor(["kn", "gb", "la", "ls", "rr"])

################################################################################
# Test 1: Check magic methods
## regressor retrieval
def test_regressor_retrieval(ensemble_regressor):
    """Test out-of-index error in combination with __getitem__ method."""
    # Arrange
    max_index = len(ensemble_regressor.regressors)
    # Assert
    with pytest.raises(IndexError): # True if IndexError is raised
        ensemble_regressor[max_index]

## regressor input
@pytest.mark.parametrize("input, expected_output",
                         [(["kn", "gb", "la"], True),
                          (["kn", "ls", "rr"], True),
                          (["kn", "abc", "la"], False), # should raise KeyError as "abc" is non-defined
                          (["kn", "kn", "la"], True)
                          ])

def test_regressor_types(input, expected_output):
    """Test EnsembleRegressor generation in combination with __init__ method."""
    if expected_output:
        for name_tag in input:
            assert name_tag in class_dict
    else:
        with pytest.raises(KeyError):
            EnsembleRegressor(input)

## regressor replacement
def test_regressor_replacement(ensemble_regressor):
    """Test replacement of regressor EnsembleRegressor in combination with __setitem__ method."""
    # Arrange
    grad_boost_regressor = GradBoostRegressor()
    # Act
    ensemble_regressor[0] = grad_boost_regressor
    # Assert
    assert ensemble_regressor[0].type == "gb"

## regressor removal
def test_regressor_removal(ensemble_regressor):
    """Test removal of regressor in combination with __delitem__ method."""
    # Arrange
    max_index = len(ensemble_regressor.regressors)
    # Assert
    with pytest.raises(IndexError):
        del ensemble_regressor[max_index]

################################################################################
# Test 2: Check single regressor protocol funtionality
@pytest.mark.parametrize("input, expected_output",
                         [(RidgeRegressionRegressor(), True),
                          (ensemble_regressor, False)
                          ])

def test_regressor_adherence(input, expected_output):
    """Test protocol definition."""
    assert isinstance(input, RegressorBase) == expected_output

################################################################################
# Test 3: Invalid input to EnsembleRegressor (empty list, non-strings, non-list)
@pytest.mark.parametrize("input, expected_output", [
    (["gb", "kn"], True),
    ([], False),          # empty list raises ValueError
    (["gb", 123], False), # non-string ValueError
    (123, False)          # non-list raises ValueError
])

def test_regressor_input(input, expected_output):
    """Test inconvenient input for EnsembleRegressor instantiation."""
    if expected_output:
        assert isinstance(EnsembleRegressor(input), EnsembleRegressor) == expected_output
    else:
        with pytest.raises(ValueError):
            EnsembleRegressor(input)

################################################################################
# Test 4: Check accuracy
def test_regressor_accuracy():
    # Arrange
    data_feature_matrix = np.array([[1, 2, 3], [2, 3, 4]]).reshape(3, 2)
    data_target = np.array([1, 2, 3])

    # Act
    # preds with single regressors
    grad_preds = (GradBoostRegressor()
                  .fit_predict(data_feature_matrix, data_target))
    rr_preds = (RidgeRegressionRegressor()
                .fit_predict(data_feature_matrix, data_target))
    regressors_preds = np.mean(np.column_stack((grad_preds, rr_preds)), axis=1)

    # preds with ensemble regrssor
    ensemble_preds = (EnsembleRegressor(["gb", "rr"])
                      .fit_predict(data_feature_matrix, data_target))

    print(regressors_preds)
    print(ensemble_preds)

    # Assert
    assert regressors_preds.all() == ensemble_preds.all()