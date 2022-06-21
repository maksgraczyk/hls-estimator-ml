import numpy as np
import hls4ml
import re
import sys
import io
import contextlib
from .. import BaseEstimation
from hls4ml.converters import convert_from_keras_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MaxPooling2D, Activation, ReLU, Softmax, \
    Flatten
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

            for x in self.metrics[device]:
                K.clear_session()
                result[x] = load(path / layer_type / x)

            return result

        if device not in ['asic', 'fpga']:
            raise RuntimeError('device must be either "asic" or "fpga"')

        self._layer_types = ['Conv2D', 'MaxPooling2D', 'BatchNormalization',
                             'Dense', 'ReLU', 'Softmax']
        
        path = Path(__file__).parent / 'single_models' / device

        self._models = [get_model(path, l) for l in self._layer_types]
        self._device = device

    @property
    def metrics(self):
        return {
            'asic': ['latency', 'static_power', 'dynamic_power', 'area'],
            'fpga': ['latency', 'lut', 'ff', 'dsp', 'dynamic_power']
        }

    @property
    def units(self):
        return {
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

        result = {x: 0 for x in self.metrics[self._device]}
        null_parameter_shares = \
            self._get_null_parameter_shares(model, conv_dense_names)

        for i, layers in enumerate(layers_by_type):
            data[i] = [[clock_frequency] +
                       self._get_layer_data(layer, null_parameter_shares)
                       for layer in layers]

        for i in range(len(data)):
            if len(data[i]) == 0:
                continue

            models = self._models[i]

            for metric in result.keys():
                model = models[metric]
                if model is None:
                    continue

                predicted = model.predict(np.array(data[i]), verbose=0)

                layer_type = self._layer_types[i]
                mean, std = \
                    self._denorm_values[self._device][layer_type][metric]
                predicted = (predicted * std) + mean

                result[metric] += np.sum(predicted)
        
        for key, value in result.items():
            result[key] = (value, self.units[self._device][key])

        return result

    @property
    def _denorm_values(self):
        # Denormalisation values in form of (mean, std)
        return {
            'asic': {
                'Conv2D': {
                    'latency': (11.983796296296296, 6.0135108368324595),
                    'static_power': (1101.7093240740742, 1042.2271158405176),
                    'dynamic_power': (9733.262412037036, 11498.909028900627),
                    'area': (47556.48785699915, 45022.00946397084)
                },

                'MaxPooling2D': {
                    'latency': (15.729166666666666, 5.365156476186097),
                    'static_power': (542.4211979166666, 564.3565553101994),
                    'dynamic_power': (5197.198125, 7953.824331578362),
                    'area': (17363.28441397349, 19505.551659748806)
                },

                'BatchNormalization': {
                    'latency': (11.666666666666666, 6.252399190470667),
                    'static_power': (349.09130208333335, 258.55263566899623),
                    'dynamic_power': (3429.736770833333, 3408.22695729471),
                    'area': (14788.143821716309, 10876.457377970793)
                },

                'Dense': {
                    'latency': (12.058333333333334, 5.958286795730159),
                    'static_power': (1812.3571166666668, 3339.7476801813314),
                    'dynamic_power': (16742.46973333333, 35692.11242679985),
                    'area': (83765.3372167015, 150686.05624703845)
                },

                'ReLU': {
                    'latency': (11.666666666666666, 6.25617974056345),
                    'static_power': (78.06141025641024, 75.52456104194842),
                    'dynamic_power': (960.0777564102565, 1185.5016544571924),
                    'area': (3521.1425543565015, 3659.92692818571)
                },

                'Softmax': {
                    'latency': (13.076923076923077, 5.080326426440557),
                    'static_power': (2682.8353846153846, 2503.5801154439932),
                    'dynamic_power': (14352.25641025641, 18741.727326299333),
                    'area': (129227.42028182592, 122334.95326464878)
                }
            },

            'fpga': {
                'Conv2D': {
                    'latency': (12.654320987654321, 5.48810416508885),
                    'lut': (5352.7390946502055, 4232.007428713101),
                    'ff': (534.2362139917695, 875.4978377622335),
                    'dsp': (61.01728395061728, 116.49248774499999),
                    'dynamic_power': (0.10693415637860064, 0.1312336809734344)
                },

                'MaxPooling2D': {
                    'latency': (19.583333333333332, 9.6987791822748),
                    'lut': (3648.1614583333335, 3692.3845443032315),
                    'ff': (1658.5, 2669.2571745476953),
                    'dynamic_power': (0.07095312499999983, 0.10640770594966893)
                },

                'BatchNormalization': {
                    'latency': (11.666666666666666, 6.252399190470667),
                    'lut': (456.40625, 353.03002788573843),
                    'ff': (681.78125, 503.215310014495),
                    'dsp': (40.359375, 29.501814934408422),
                    'dynamic_power': (0.02099479166666647, 0.02118368862664657)
                },

                'Dense': {
                    'latency': (12.775, 5.622934242123297),
                    'lut': (9546.573333333334, 15602.34117394071),
                    'ff': (1163.3275, 2978.9257248162567),
                    'dsp': (172.885, 456.3794641389332),
                    'dynamic_power': (0.24526666666666658, 0.5336967216636095)
                },

                'ReLU': {
                    'latency': (11.666666666666666, 6.25617974056345),
                    'lut': (545.6153846153846, 504.33745054264125),
                    'ff': (562.6346153846154, 553.7837249503195),
                    'dynamic_power': (0.003993589743589633, 0.005217213191179378)
                },

                'Softmax': {
                    'latency': (21.923076923076923, 5.208215248323987),
                    'lut': (6145.871794871795, 5872.558664892551),
                    'ff': (900.6153846153846, 2085.272361554324),
                    'dsp': (68.07692307692308, 63.661268327140974),
                    'dynamic_power': (0.0657692307692306, 0.09270708300268324)
                }
            }
        }

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
        elif isinstance(layer, Activation) and layer.activation == softmax:
            index = 5

        if index is None and not isinstance(layer, Flatten):
            print(f'Layer f{layer.name} of type {type(layer)} not supported, '
                  'ignoring',
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
              layer.activation == softmax):
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
        result = {}

        with TemporaryDirectory() as tmp_dir:
            with contextlib.redirect_stdout(io.StringIO()):
                config = \
                    hls4ml.utils.config_from_keras_model(model=model,
                                                         granularity='name')

                convert_from_keras_model(model, hls_config=config,
                                         output_dir=tmp_dir).write()

            tmp_path = Path(tmp_dir) / 'firmware' / 'weights'
            weight_file_paths = list(filter(lambda x: x.match('w*.txt'),
                                            tmp_path.iterdir()))
            weight_file_paths.sort(key=lambda x: int(x.stem[1:]))

            for i, weight_file_path in enumerate(weight_file_paths):
                weights = []
                biases = []

                n = re.search(r'w(\d+)\.txt', str(weight_file_path)).group(1)
                
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
