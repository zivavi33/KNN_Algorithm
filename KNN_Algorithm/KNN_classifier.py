import numpy as np
from collections import Counter

#! replace validation to dataclass or pydentic
#? validation: in knn algorithm there is an important hyper-parameter: `K` -> how many nearest neighbours to calculate distance from.
#? we need to validate that the value is int and not float or string and positive value of >= 1.

#* Exception for value must be => 1 error. 
class ValueToSmallError(Exception):
    def __init__(self,messege,value):
        self.messege = messege
        self.value = value
#* Exception for value must be from type 'int' error.
class ValueMustBeIntError(Exception):
    def __init__(self,messege,value):
        self.messege = messege
        self.value = value

#* validation function
def _value_test(value_to_check):
    """this function will check the input value for `k`
        and raise value error if needed """
    if not isinstance(value_to_check, int):
        raise ValueMustBeIntError("value must be an integer",value_to_check)
    elif value_to_check < 1 :
        raise ValueToSmallError("value must be greater than one",value_to_check)
    

#* the algorithm class
class Knn_algorithm:
    def __init__(self, k=3) -> None:
        self._k = k


    @property
    def _k(self) -> int:
        return self.__k
    @_k.setter
    def _k(self,value:int) -> None:
        _value_test(value)
        self.__k = value


    def fit(self,X:np.ndarray, Y:np.ndarray) -> None:
        self.X_train = X
        self.Y_train = Y   

    def _inter_predict(self,x:np.ndarray) -> np.int64:
        """helper function to `predict` function. this function calculates the distances and returns the most common class in the `k` closest vectors"""
        all_distances = [self._euclidean_distance(x,x_train)for x_train in self.X_train]
        # get the k nearest neighbors classification
        idxs = np.argsort(all_distances)[:self._k]
        labels = [self.Y_train[i] for i in idxs] 
        # get the most common label
        most_common = Counter(labels).most_common(1)
        return most_common[0][0]
    
    def predict(self,X:np.ndarray) -> np.ndarray:
        """this function applies the _inter_predict helper on all of the data points and returns vector of predictions"""
        return np.array([self._inter_predict(x) for x in X])
        
    
    @staticmethod
    def accuracy(y_true,y_pred):
        return np.sum(y_true == y_pred) / len(y_true)
    @staticmethod
    def _manhattan_distance(x1:np.ndarray, x2:np.ndarray) -> float:
        """calculate manhattan distance between x1 and x2 vectors"""
        #! the buttom line is more fast and more efficient but this code is for learning purposes only:
        # return np.sum(np.abs(np.subtract(x1 , x2)))
        abs_delta = np.absolute(np.subtract(x1 , x2))
        elemnts_sum = np.sum(abs_delta)
        return elemnts_sum 
    @staticmethod
    def _euclidean_distance(x1:np.ndarray, x2:np.ndarray) -> float:
        """calculate euclidean distance between x1 and x2 vectors"""
        #! the buttom line is more fast and more efficient but this code is for learning purposes only:
        #return np.sqrt(np.sum((np.subtract(x1,x2))**2))
        delta = np.subtract(x1 , x2)
        elemnts_sum = np.sum((delta ** 2))
        return np.sqrt(elemnts_sum)