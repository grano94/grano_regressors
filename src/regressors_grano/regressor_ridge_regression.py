from sklearn.linear_model import Ridge      # ridge regression regressor
import numpy as np


class RidgeRegressionRegressor:
    """
    Ridge regression regressor class
    """

    def __init__(self, name: str="Ridge Regression Regressor"):
        """Instantiates ridge regressor.

        Args:
            name (str, optional): Name tag for regessor. Defaults to "Ridge Regression Regressor".
        """
        self.name = name
        self.type = "rr"
        self.model = Ridge()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the regression model.

        Args:
            X (np.ndarray): feature matrix
            y (np.ndarray): target array
        """
        self.model.fit(X, y)
        print("Regression model trained.")
    
    def predict(self, X: np.ndarray):
        """Predict using the trained regression model.

        Args:
            X (np.ndarray): feature matrix

        Returns:
            np.ndarray: Predictions of regressor.
        """
        predictions = self.model.predict(X)
        print("Regression model predictions provided.")
        return predictions        
       
    def fit_predict(self, X: np.ndarray, y: np.ndarray):  
        """Train the regression model and return predictions for the training data.

        Args:
            X (np.ndarray): feature matrix
            y (np.ndarray): target array

        Returns:
            np.ndarray: Predictions of regressor.
        """
        self.model.fit(X, y)
        predictions = self.model.predict(X) 
        print("Regression model trained and predictions provided.")
        return predictions 
 
    def score(self, X: np.ndarray, y: np.ndarray):   
        """Calculate the r2 score of the model.
        
        Args:
            X (np.ndarray): feature matrix
            y (np.ndarray): target array

        Returns:
            float: R2 score.
        """
        score = self.model.score(X, y)
        print(f"{self.name} regressor r2_score = {score}") 
        return score  




