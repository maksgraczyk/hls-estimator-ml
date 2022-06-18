import numpy as np
import hls4ml
import re
import sys
from . import BaseEstimation
from hls4ml.converters import convert_from_keras_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MaxPooling2D, Activation, ReLU, Softmax
from tensorflow.keras.activations import softmax
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qlayers import QDense, QActivation
from qkeras.qnormalization import QBatchNormalization
from qkeras.qconvolutional import QConv2D
from pathlib import Path
from tempfile import TemporaryDirectory


class SingleOutputEstimation(BaseEstimation):
    """
    A class representing single-output regression for estimating hardware
    metrics for neural network models. It deploys one multi-layer perceptron
    per layer type per hardware metric.

    QConv2D, QBatchNormalization, QDense, and quantised ReLU from QKeras
    are supported, along with Softmax from Keras. Keras Activation objects
    are also supported if they are set for Softmax.

    If QKeras layers are used, they must be quantised with the quantized_bits
    quantiser or quantised_relu in case of ReLU.

    This approximation class works best for Xilinx xcvu9p-flgb2104-2L-e FPGAs
    and ASICs implemented in the 45 nm technology (ideally Nangate).

    Args:
    ----------
    device : str, either 'asic' or 'fpga'
       The device type for approximation.
    """

    def __init__(self, device):
        def load(path):
            return load_model(path) if path.exists() else None

        def get_model(path, layer_type):
            result = {}

            for x in self.METRICS[device]:
                K.clear_session()
                result[x] = load(path / layer_type / x)

            return result

        if device not in ['asic', 'fpga']:
            raise RuntimeError('device must be either "asic" or "fpga"')

        layer_types = ['Conv2D', 'MaxPooling2D', 'BatchNormalization',
                       'Dense', 'ReLU', 'Softmax']
        
        path = Path(__file__).parent / 'single_models' / device

        self._models = [get_model(path, l) for l in layer_types]
        self._device = device

    def _get_model_index(self, layer):
        index = None
        
        if isinstance(layer, QConv2D):
            index = 0
        elif isinstance(layer, MaxPooling2D):
            index = 1
        elif isinstance(layer, QBatchNormalization):
            index = 2
        elif isinstance(layer, QDense):
            index = 3
        elif isinstance(layer, ReLU) or \
             (isinstance(layer, QActivation) and isinstance(layer.activation,
                                                            quantized_relu)):
            index = 4
        elif isinstance(layer, Softmax):
            index = 5
        elif isinstance(layer, Activation) and isinstance(layer.activation,
                                                          softmax):
            index = 5

        if index is None:
            print(f'Layer type {type(layer)} not supported, ignoring',
                  file=sys.stderr)

        return index

    def _get_layer_data(self, layer, null_parameter_shares=None):
        layer_data = []
        
        if isinstance(layer, QConv2D):
            output_area = 1

            for num in layer.output_shape:
                if num is not None:
                    output_area *= num

            input_channels = layer.input_shape[-1]

            kernel_area = 1

            for num in layer.kernel_size:
                kernel_area *= num

            if not isinstance(layer.kernel_quantizer, quantized_bits) or \
               not isinstance(layer.bias_quantizer, quantized_bits):
                raise NotImplementedError(f'Layer {layer.name} must be '
                                          'quantised with quantized_bits')
                
            kernel_bitwidth = layer.kernel_quantizer.bits
            bias_bitwidth = layer.bias_quantizer.bits

            if null_parameter_shares is None:
                raise RuntimeError(f'Layer {layer.name} is QConv2D, so'
                                   'null_parameter_shares must be provided')
            
            null_parameter_share = null_parameter_shares[layer.name]

            layer_data = [output_area,
                          input_channels,
                          kernel_area,
                          kernel_bitwidth,
                          bias_bitwidth,
                          null_parameter_share]
        elif isinstance(layer, MaxPooling2D):
            output_area = 1

            for num in layer.output_shape[:-1]:
                if num is not None:
                    output_area *= num

            input_channels = layer.input_shape[-1]

            kernel_area = 1

            for num in layer.pool_size:
                kernel_area *= num

            layer_data = [output_area,
                          input_channels,
                          kernel_area]
        elif isinstance(layer, QBatchNormalization):
            inputs = 1

            for num in layer.input_shape:
                if num is not None:
                    inputs *= num

            if not isinstance(layer.beta_quantizer, quantized_bits) or \
               not isinstance(layer.gamma_quantizer, quantized_bits):
                raise NotImplementedError(f'Layer {layer.name} must be '
                                          'quantised with quantized_bits')

            beta_bitwidth = layer.beta_quantizer.bits
            gamma_bitwidth = layer.gamma_quantizer.bits

            layer_data = [inputs,
                          beta_bitwidth,
                          gamma_bitwidth]
        elif isinstance(layer, QDense):
            inputs = 1

            for num in layer.input_shape:
                if num is not None:
                    inputs *= num

            outputs = 1

            for num in layer.output_shape:
                if num is not None:
                    outputs *= num

            if not isinstance(layer.kernel_quantizer, quantized_bits) or \
               not isinstance(layer.bias_quantizer, quantized_bits):
                raise NotImplementedError(f'Layer {layer.name} must be '
                                          'quantised with quantized_bits')

            weight_bitwidth = layer.kernel_quantizer.bits
            bias_bitwidth = layer.bias_quantizer.bits

            if null_parameter_shares is None:
                raise RuntimeError(f'Layer {layer.name} is QDense, so'
                                   'null_parameter_shares must be provided')
            
            null_parameter_share = null_parameter_shares[layer.name]

            layer_data = [inputs,
                          outputs,
                          weight_bitwidth,
                          bias_bitwidth,
                          null_parameter_share]
        elif isinstance(layer, QActivation):
            if not isinstance(layer.activation, quantized_relu):
                raise NotImplementedError(f'Activation {layer.name} is '
                                          'QActivation, so it must have '
                                          'quantized_relu')

            inputs = 1

            for num in layer.input_shape:
                if num is not None:
                    inputs *= num

            bitwidth = layer.activation.bits

            layer_data = [inputs,
                          bitwidth]
        elif isinstance(layer, Softmax) or \
             (isinstance(layer, Activation) and \
              isinstance(layer.activation, softmax)):
            inputs = 1

            for num in layer.input_shape:
                if num is not None:
                    inputs *= num

            layer_data = [inputs]
        else:
            raise NotImplementedError(f'Layer type {type(layer)} '
                                      'not supported')

        return layer_data

    def _get_null_parameter_shares(self, model, layer_names):
        config = hls4ml.utils.config_from_keras_model(model=model,
                                                      granularity='name')
        result = {}

        with TemporaryDirectory() as tmp_dir:
            convert_from_keras_model(model, hls_config=config,
                                     output_dir=tmp_dir).write()

            tmp_path = Path(tmp_dir) / 'firmware' / 'weights'
            weight_file_paths = list(filter(lambda x: x.match('w*.txt'),
                                            tmp_path.iterdir()))
            
            for i, weight_file_path in enumerate(weight_file_paths):
                weights = []
                biases = []

                n = \
                    re.search(r'w(\d+)\.txt', str(weight_file_path)).group(1)
                
                with weight_file_path.open(mode='r') as f:
                    for line in f:
                        weights += line.strip().split(', ')

                with (weight_file_path.parent / f'b{n}.txt').open(
                        mode='r') as f:
                    for line in f:
                        biases += line.strip().split(', ')

                if len(weights) > 0:
                    weight_zeros = len(
                        list(filter(lambda x: re.search(r'^0*(\.0+)?$', x)
                                    or re.search(r'^nan$', x), weights)))
                else:
                    weight_zeros = 0
                    
                if len(biases) > 0:
                    bias_zeros = len(
                        list(filter(lambda x: re.search(r'^0*(\.0+)?$', x)
                                    or re.search(r'^nan$', x), biases)))
                else:
                    bias_zeros = 0

                if len(weights) + len(biases) > 0:
                    share = (weight_zeros + bias_zeros) / \
                        (len(weights) + len(biases))
                else:
                    raise RuntimeError(f'{weight_file_path} and b*.txt do not '
                                       'have any non-NaN weights or biases')

                result[layer_names[i]] = share
        
        return result
    
    def predict(self, model, clock_frequency):
        super().predict(model, clock_frequency)

        layers_by_type = [
            [],  # Conv2D
            [],  # MaxPooling2D
            [],  # BatchNormalization
            [],  # Dense
            [],  # ReLU
            []   # Softmax
        ]

        conv_dense_names = []

        for layer in model.layers:
            index = self._get_model_index(layer)

            if index is not None:
                layers_by_type[self._get_model_index(layer)].append(layer)

                if type(layer) in [QConv2D, QDense]:
                    conv_dense_names.append(layer.name)

        data = [
            [],  # Conv2D
            [],  # MaxPooling2D
            [],  # BatchNormalization
            [],  # Dense
            [],  # ReLU
            []   # Softmax
        ]

        result = {x: 0 for x in self.METRICS[self._device]}
        null_parameter_shares = \
            self._get_null_parameter_shares(model, conv_dense_names)

        for i, layers in enumerate(layers_by_type):
            data[i] = [[clock_frequency] +
                       self._get_layer_data(layer, null_parameter_shares)
                       for layer in layers]

        for i in range(len(data)):
            models = self._models[i]

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
                'dynamic_power': 'W'
            }
        }
        
        for key, value in result.items():
            result[key] = (value, units[self._device][key])

        return result
