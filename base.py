from abc import ABC, abstractmethod


class BaseEstimation(ABC):
    METRICS = {
        'asic': ['latency', 'static_power', 'dynamic_power', 'area'],
        'fpga': ['latency', 'lut', 'ff', 'dsp', 'dynamic_power']
    }
    
    @abstractmethod
    def predict(self, model, clock_frequency, device):
        """
        Predicts hardware metrics for a given neural network model, clock
        frequency, and device type.

        Args:
        ----------
        model : keras.Model
           A neural network model for which the estimation should be made.

        clock_frequency : int or float
           The clock frequency in MHz.

        device : str, either 'asic' or 'fpga'
           The device type for which the estimation should be made (i.e.
           either ASIC or FPGA).

        Returns:
        ----------
        A dictionary containing the predicted hardware metrics for the
        given model. Every dictionary value is either None (i.e. not supported)
        or of form (X, Y), where X is the predicted value and Y is a string
        with the unit used.
        """
        pass
