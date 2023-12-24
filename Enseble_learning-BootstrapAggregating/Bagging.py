import numpy as np
from collections import Counter
import multiprocessing
from multiprocessing import Process , Queue
import time
import argparse

# class ValueMustBeInteger(Exception):
#     def __init__(self,messege,value):
#         self.messege = messege
#         self.value = value

# class ValueMustBeFloat(Exception):
#     def __init__(self,messege,value):
#         self.messege = messege
#         self.value = value

multiprocessing.set_start_method('fork') #'spawn'



class Bagging:
    def __init__(self,model:object, sample_size:float = 1.0, number_bags:int = 2, replacement:bool = True) -> None:
        self._model = model
        self._part_to_sample = sample_size
        self._number_bags = number_bags
        self._replacement = replacement
        self._bags : None | list = None

    @property
    def _part_to_sample(self):
        return self.__part_to_sample
    @_part_to_sample.setter
    def _part_to_sample(self,value):
        if  value > 0.0  and value <=1.0 and isinstance(value, float):
            self.__part_to_sample = value
        else:
            raise ValueError(f"'_part_to_sample' must be a float > 0 and  <= 1 ({value})")
            

    @property
    def _number_bags(self):
        return self.__number_bags
    @_number_bags.setter
    def _number_bags(self,value):
        if  value > 1  and isinstance(value, int):  
            self.__number_bags = value
        elif value == 1:
            raise Warning(f"_number_bags value = ({value}). it is not recommended because it is not actually bagging anymore ")
        else:
            raise ValueError(f"1) 'number_bags' must be a positive integer greater than '0'! ({value}) ")

    @property
    def _replacement(self):
        return self.__replacement
    @_replacement.setter
    def _replacement(self,value):
        if  isinstance(value, bool):
            self.__replacement = value
        else:
            raise ValueError(f"'_replacement' must be of a boolean type ({value})")


    def bootstrap_sample(self,X,y):
        """ function that take 1 sample/bag from data with or without repetitions"""
        samples_num = X.shape[0]
        size = int(self._part_to_sample * samples_num)
        indexs = np.random.choice(samples_num, size = size, replace = self._replacement)
        return X[indexs], y[indexs]
    
    def _fit(self,X,y,q):
        X_sample , y_sample = self.bootstrap_sample(X,y)
        bag = self._model.fit(X_sample , y_sample)
        q.put(bag)
        print("weak learner: ",str(bag.score(X_sample , y_sample)))

    def fit(self,X,y):
        """ fit the model to the different data sets- bags"""
        queue = Queue()
        self._bags = []
        for _ in range(self._number_bags):
            p = Process(target= self._fit, args=(X,y,queue))
            p.start()
            print(f" procces: {p.name}")
        for _ in range(self._number_bags):
            x = queue.get()
            self._bags.append(x)
        print(self._bags)

    def _most_common_label(self,y):
        """ democratic simple hard voting from all of the weak learners"""
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self,X):
        """make predictions from each weak learner and rearrange the data so we get predictions from all of the weak 
        learners for each data point"""
        weak_predictions = np.array([bag.predict(X) for bag in self._bags])
        weak_predictions = np.swapaxes(weak_predictions, 0,1)
        y_pred = [self._most_common_label(bag_pred) for bag_pred in weak_predictions]
        return np.array(y_pred)


    @ staticmethod
    def accuracy(y_true,y_pred):
        """ check accuracy of bagging model"""
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy