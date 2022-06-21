# HLSEstimatorML

A machine-learning-based package for estimating all important hardware metrics of any given neural network as implemented by a high-level-synthesis-based machine learning framework (HLS-based ML framework), with little hardware expertise and few changes to HLS-based ML frameworks required.

## What is an HLS-based ML framework?
An HLS-based ML framework raises the abstraction level of implementing neural networks on FPGAs/ASICs and therefore bridges the gap between ML developers and hardware acceleration.

In general, it expects a machine learning model defined with a standard ML framework like PyTorch or TensorFlow/Keras along with some basic hardware parameters (understandable for a non-hardware person) and target device. Afterwards, it produces an HLS code which is then synthesised automatically into an RTL code and then into a bitstream/netlist ready to be deployed to an FPGA or ASIC. An HLS-based ML framework usually uses a specific set of HLS and RTL tools.

See the figure below for the graphical explanation. The details may vary depending on the specific framework.

![A diagram showing how HLS-based ML frameworks work in general.](https://user-images.githubusercontent.com/24892582/174791618-43382027-2cfb-47d7-8193-1954bcb318f3.svg)


## Supported HLS-based ML frameworks and devices
* [hls4ml](https://github.com/fastmachinelearning/hls4ml) with [the Catapult HLS backend](https://github.com/fastmachinelearning/hls4ml-catapult-framework) *(a multi-layer perceptron is deployed independently per metric per layer type, no non-SRAM memory supported because of the lack of such support in the backend)*
  * FPGAs: Xilinx Virtex UltraScale+ xcvu9p-flgb2104-2L-e
  * ASICs: Nangate 45 nm technology

## Estimated outputs
* FPGAs: latency, LUT count, FF count, DSP count, dynamic power
* ASICs: latency, static power, dynamic power, silicon area

## How to use this tool?
No PyPI version is available at the moment, so the installation has to be done manually. Clone this repository and run the following command inside:
```
pip install -e .
```

Afterwards, you can start producing estimates by importing relevant `hlsestimatorml` modules. Here's an example for a one-layer perceptron implemented by hls4ml for 200 MHz ASICs:
```python3
from tensorflow.keras.models import Sequential
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from hlsestimatorml.hls4ml import CatapultSingleOutputEstimation

model = Sequential()
model.add(QDense(32, input_shape=(16,), kernel_quantizer=quantized_bits(16, 5),
          bias_quantizer=quantized_bits(8, 4)))
model.add(QActivation(activation=quantized_relu(9, 7)))

estimation = CatapultSingleOutputEstimation(device='asic')
print(estimation.predict(model, clock_frequency=200))
```

After running the above code, you should see a printed Python dictionary similar to this one:
```python3
{
  'latency': (10.680241107940674, 'ns'),
  'static_power': (1552.4939575195312, 'uW'),
  'dynamic_power': (16617.88885498047, 'uW'),
  'area': (66309.44360351562, 'um^2')
}
```

Every tuple is of form (X, Y), where X is the predicted value and Y is the unit used.

If you want to know more details, you can have a look at the docstrings, e.g. in `base.py`.

## Future work
HLSEstimatorML is built with extensibility in mind. Here are some points describing the potential future work:
* Generalising HLSEstimatorML so that no separate training is required for every extra FPGA device or ASIC technology.
* Adding support of other hls4ml backends and other HLS-based ML frameworks.
* Including different machine learning algorithms and more accurate models for the estimation.

## License
See LICENSE.
