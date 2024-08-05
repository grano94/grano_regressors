from typing import Protocol, runtime_checkable

@runtime_checkable # decorater required to make class RegressorBase runtime-checkable
class RegressorBase(Protocol):
    """ 
    Base class used to specify methods / attributes required by regressor classes.
    """

    name: str                       # required attribute


    type: str                       # required attribute


    def fit(self, X, y):            # required method
        ...
    
    def predict(self, X):           # required method
        ...

    def fit_predict(self, X, y):    # required method
        ...

    def score(self, X, y):          # required method
        ...