from tensorflow.keras import Model
from abc import ABC, abstractmethod


class BaseEstimation(ABC):
    METRICS = {
        'asic': ['latency', 'static_power', 'dynamic_power', 'area'],
        'fpga': ['latency', 'lut', 'ff', 'dsp', 'dynamic_power']
    }
    
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
