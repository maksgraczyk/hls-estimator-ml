from tensorflow.keras import Model
from abc import ABC, abstractmethod


class BaseEstimation(ABC):
    @abstractmethod
    def metrics(self):
        """
        Metrics to predict for each device type, in form of the following dict:
        {'asic': (list of strings representing metrics),
         'fpga': (list of strings representing metrics)}
        """
        pass

    @abstractmethod
    def units(self):
        """
        Units for each metric, in form of the following dict:
        {'asic': {(metric in str): (unit in str)},
         'fpga': {(metric in str): (unit in str)}}
        """
        pass
    
    @abstractmethod
    def predict(self, model, clock_frequency):
        """
        Predicts hardware metrics for a given neural network model and clock
        frequency.

        Args:
        ----------
        model : keras.Model
           A neural network model for which the estimation should be made.

        clock_frequency : int or float
           The clock frequency in MHz.

        Returns:
        ----------
        A dictionary containing the predicted hardware metrics for the
        given model. Every dictionary value is either None (i.e. not supported)
        or of form (X, Y), where X is the predicted value and Y is a string
        with the unit used.
        """
        if not isinstance(model, Model):
            raise RuntimeError('model is not an instance of keras.Model')
