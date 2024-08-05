from .regressor_base import RegressorBase                # defines the base class for the regressor types
from .regressor_grad_boost import GradBoostRegressor
from .regressor_lasso import LassoRegressor
from .regressor_nearest_neighbor import KNeigRegressor
from .regressor_ordinary_least_squares import OrdinaryLeastSquaresRegressor
from .regressor_ridge_regression import RidgeRegressionRegressor
import numpy as np
from sklearn.metrics import r2_score


# The ensemble regressor can handle the following keys
class_dict = {"kn": KNeigRegressor(),
              "gb": GradBoostRegressor(),
              "la": LassoRegressor(),
              "ls": OrdinaryLeastSquaresRegressor(),
              "rr": RidgeRegressionRegressor()}


class EnsembleRegressor:
    """
    Ensemble regressor container object
    """

    def __init__(self, regressor_list: list=["ls"]):
        """Instantiate multi-model regressor object

        Args:
            regressor_list (list, optional): List with regression model types ("kn", "gb", "la", "ls", "rr") provided as strings. Defaults to ["ls"].

        Raises:
            KeyError: Key error raised if provided regressor name is non-defined.
        """
        self.regressors = []

        if not isinstance(regressor_list, list) or not regressor_list:
            raise ValueError("Input should be a non-empty list.")
        if not all(isinstance(item, str) for item in regressor_list):
            raise ValueError("All elements in the list should be strings.")
        
        for regressor_key in regressor_list:
            if regressor_key not in class_dict:
                # If KeyError is raised, the object will not be created.
                raise KeyError(f"Regressor key '{regressor_key}' not defined.")
            else:
                print(f"{class_dict[regressor_key].name} loaded.")
                self.regressors.append(class_dict[regressor_key])
    
    def __getitem__(self, index: int) -> RegressorBase:
        """Retrieve regressor from multi-model regressor container object.

        Args:
            index (int): Index of regressor that should be returned.

        Raises:
            IndexError: Error raised if index is out or range.

        Returns:
            RegressorBase: Regressor at given index.
        """
        if index < 0 or index >= (len(self.regressors)):
             raise IndexError(f"Provided regressor key must be 0 <= index < {len(self.regressors)}")
        
        print(f"{self.regressors[index].name} returned.")
        return self.regressors[index]
    
    def __setitem__(self, index: int, regressor):#: RegressorBase):
        """Replace regressor at given index.

        Args:
            index (int): Index of regressor that should be changed.
            regressor (RegressorBase): Regressor type that should be included.

        Raises:
            TypeError: Error raised if given regressor is not adhering to RegressorBase protocol.
        """
        if not (isinstance(regressor, RegressorBase)):
            raise TypeError("Provided regressor does not implement necessary attributes and methods defined in RegressorBase!")
        else:
            print(f"{self.regressors[index].name} replaced by {regressor.name}.")
            self.regressors[index] = regressor

    def __delitem__(self, index: int):
        """Remove regression model from multi-model regressor.

        Args:
            index (int): Index of regressor that should be removed.

        Raises:
            IndexError: Error rasied if index is out of range.
        """
        if index < 0 or index >= len(self.regressors):
            raise IndexError(f"Provided regressor key must be 0 <= key < {len(self.regressors)}") 
        else:
            del self.regressors[index]

    def __str__(self) -> str:
        """Show regression types that are loaded.

        Returns:
            str: Regression models that are loaded.
        """
        regressor_names = ", ".join([f"{regressor.name}" for regressor in self.regressors])
        return f"The following regressors are included: {regressor_names}"
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the multi-model regressors.

        Args:
            X (np.ndarray): feature matrix
            y (np.ndarray): target array
        """
        for regressor in self.regressors:
            print(f"Training {regressor.name}:")
            regressor.fit(X, y)   
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained multi-model regressor.

        Args:
            X (np.ndarray): feature matrix

        Returns:
            np.ndarray: Predictions representing mean of all regressors.
        """
        accumulated_preds = None

        # retrieve predictions from distinct regressors
        for regressor in self.regressors:
            print(f"{regressor.name}:")
            if accumulated_preds is None:
                accumulated_preds = regressor.predict(X).reshape(-1, 1) # reshaped to be a 2D array
            else:
                new_preds = regressor.predict(X).reshape(-1, 1)
                accumulated_preds = np.column_stack((accumulated_preds, new_preds))

        # return averaged predictions
        return np.mean(accumulated_preds, axis=1)

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:  
        """Train the multi-model regressor and return predictions for the training data.

        Args:
            X (np.ndarray): feature matrix
            y (np.ndarray): target array

        Returns:
            np.ndarray: Predictions representing mean of all regressors.
        """
        self.fit(X, y) 
        return self.predict(X)
 
    def score(self, X: np.ndarray, y: np.ndarray) -> float:   
        """Calculate the r2 score of the multi-model regressor.
        
        Args:
            X (np.ndarray): feature matrix
            y (np.ndarray): target array
        
        Returns:
            float: R2 score.
        """
        y_true = y
        y_pred = self.predict(X)
        score = r2_score(y_true, y_pred)
        print(f"Ensemble regressor r2_score = {score}")
        return score





    
