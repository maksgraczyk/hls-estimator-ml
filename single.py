import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, \
    Dense, Activation, ReLU, Softmax
from tensorflow.keras.activations import relu, softmax
from pathlib import Path


class SingleOutputEstimation(BaseEstimation):
    def __init__(self):
        path_asic = Path(__file__).parent / 'single_models' / 'asic'
        path_fpga = Path(__file__).parent / 'single_models' / 'fpga'
        
        self._models = {
            'asic': [
                load_model(path_asic / 'Conv2D'),
                load_model(path_asic / 'MaxPooling2D'),
                load_model(path_asic / 'BatchNormalization'),
                load_model(path_asic / 'Dense'),
                load_model(path_asic / 'ReLU'),
                load_model(path_asic / 'Softmax')
            ],

            'fpga': [
                load_model(path_fpga / 'Conv2D'),
                load_model(path_fpga / 'MaxPooling2D'),
                load_model(path_fpga / 'BatchNormalization'),
                load_model(path_fpga / 'Dense'),
                load_model(path_fpga / 'ReLU'),
                load_model(path_fpga / 'Softmax')
            ]
        }

    def _get_model_index(self, layer):
        index = None
        
        if isinstance(layer, Conv2D):
            index = 0
        elif isinstance(layer, MaxPooling2D):
            index = 1
        elif isinstance(layer, BatchNormalization):
            index = 2
        elif isinstance(layer, Dense):
            index = 3
        elif isinstance(layer, ReLU):
            index = 4
        elif isinstance(layer, Softmax):
            index = 5
        elif isinstance(layer, Activation):
            if isinstance(layer.activation, relu):
                index = 4
            elif isinstance(layer.activation, softmax):
                index = 5

        if index is None:
            raise NotImplementedError(f'Layer type {type(layer)} not '
                                      'supported')

        return index
    
    def _get_model(self, layer, device):
        return self._models[device][self._get_model_index(layer)]
    
    def predict(self, model, clock_frequency, device):
        if not isinstance(model, keras.Model):
            raise RuntimeError('model is not an instance of keras.Model')

        if device not in ['asic', 'fpga']:
            raise RuntimeError('device must be either "asic" or "fpga"')

        layers = [
            [],  # Conv2D
            [],  # MaxPooling2D
            [],  # BatchNormalization
            [],  # Dense
            [],  # ReLU
            []   # Softmax
        ]

        all_layers = model.layers

        for layer in all_layers:
            layers[self._get_model_index(layer)].append(layer)
