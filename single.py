import tensorflow.keras as keras
import numpy as np
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
                {x: load_model(path_asic / f'Conv2D_{x}')
                 for x in self.METRICS['asic']},
                {x: load_model(path_asic / f'MaxPooling2D_{x}')
                 for x in self.METRICS['asic']},
                {x: load_model(path_asic / f'BatchNormalization_{x}')
                 for x in self.METRICS['asic']},
                {x: load_model(path_asic / f'Dense_{x}')
                 for x in self.METRICS['asic']},
                {x: load_model(path_asic / f'ReLU_{x}')
                 for x in self.METRICS['asic']},
                {x: load_model(path_asic / f'Softmax_{x}')
                 for x in self.METRICS['asic']}
            ],

            'fpga': [
                {x: load_model(path_fpga / f'Conv2D_{x}')
                 for x in self.METRICS['fpga']},
                {x: load_model(path_fpga / f'MaxPooling2D_{x}')
                 for x in self.METRICS['fpga']},
                {x: load_model(path_fpga / f'BatchNormalization_{x}')
                 for x in self.METRICS['fpga']},
                {x: load_model(path_fpga / f'Dense_{x}')
                 for x in self.METRICS['fpga']},
                {x: load_model(path_fpga / f'ReLU_{x}')
                 for x in self.METRICS['fpga']},
                {x: load_model(path_fpga / f'Softmax_{x}')
                 for x in self.METRICS['fpga']}
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

    def _get_layer_data(self, layer):
        layer_data = []
        
        if isinstance(layer, Conv2D):
            pass
        elif isinstance(layer, MaxPooling2D):
            pass
        elif isinstance(layer, BatchNormalization):
            pass
        elif isinstance(layer, Dense):
            pass
        elif isinstance(layer, ReLU):
            pass
        elif isinstance(layer, Softmax):
            pass
        else:
            pass

        return layer_data
    
    def predict(self, model, clock_frequency, device):
        if not isinstance(model, keras.Model):
            raise RuntimeError('model is not an instance of keras.Model')

        if device not in ['asic', 'fpga']:
            raise RuntimeError('device must be either "asic" or "fpga"')

        layers_by_type = [
            [],  # Conv2D
            [],  # MaxPooling2D
            [],  # BatchNormalization
            [],  # Dense
            [],  # ReLU
            []   # Softmax
        ]

        for layer in model.layers:
            layers_by_type[self._get_model_index(layer)].append(layer)

        data = [
            [],  # Conv2D
            [],  # MaxPooling2D
            [],  # BatchNormalization
            [],  # Dense
            [],  # ReLU
            []   # Softmax
        ]

        result = {x: 0 for x in self.METRICS[device]}

        for i, layers in enumerate(layers_by_type):
            data[i] = [[clock_frequency] + self._get_layer_data(layer)
                       for layer in layers]

        for i in len(data):
            models = self._models[device][i]

            for metric in result.keys():
                result[metric] += \
                    np.sum(models[metric].predict(np.array(data[i])))

        units = {
            'asic': {
                'latency': 'ns',
                'static_power': 'uW',
                'dynamic_power': 'uW',
                'area': 'um^2'
            },

            'fpga': {
                'latency': 'ns',
                'lut': '',
                'ff': '',
                'dsp': '',
                'bram': '',
                'dynamic_power': 'W'
            }
        }
        
        for key, value in result.items():
            result[key] = (value, units[device][key])

        return result
