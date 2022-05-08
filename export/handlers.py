from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from export.functions import *


class _WrapperModule(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = lambda *args, **kwargs: func(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class Handler:
    def __init__(self):
        self.args = {}


class QuantizeHandler(Handler):
    def forward_hook(self, module, inputs, outputs):
        s, z = module.calc_output_scale_and_zero_point()
        self.args[module] = OrderedDict({
            "output_scale": s,
            "ouput_zero_point": z,
            "output_dtype": torch.int8,
            "output_axis": None,
        })

    def replace_module(self, module):
        return _WrapperModule(lambda x: QuantizeLinearFn.apply(x, *self.args[module].values()))


class QuantizedConv2dBatchNorm2dReLUHandler(Handler):
    def forward_hook(self, module, inputs, outputs):
        s, z = module.calc_output_scale_and_zero_point()
        weight, bias = module._get_fused_weight_and_bias(None)
        quantized_weight = module._quantize_weight(weight)
        quantized_bias = module._quantize_bias(
            inputs[0], quantized_weight, bias)
        padding = module.conv2d.padding
        padding = (padding[0], padding[0], padding[1], padding[1])
        self.args[module] = OrderedDict({
            "input_scale": inputs[0].s.detach(),
            "input_zero_point": inputs[0].z,
            "int_weight": quantized_weight.q,
            "weight_scale": quantized_weight.s.detach(),
            "weight_zero_point": quantized_weight.z,
            "output_scale": s.detach(),
            "output_zero_point": z,
            "output_dtype": torch.int8,
            "int_bias": quantized_bias.q,
            "out_shape": tuple(outputs.q.shape),
            "kernel_size": module.conv2d.kernel_size,
            "padding": padding,
            "stride": module.conv2d.stride,
            "groups": module.conv2d.groups,
            "dilation": module.conv2d.dilation,
        })
        assert z == -128 or module.activation is None, \
            f"{module.running_min}, {module.running_max}"

    def replace_module(self, module):
        return _WrapperModule(lambda x: QLinearConvFn.apply(x, *self.args[module].values()))


class QuantizedReLUHandler(Handler):
    def forward_hook(self, module, inputs, outputs):
        s, z = module.calc_output_scale_and_zero_point()
        self.args[module] = OrderedDict({
            "input_scale": inputs[0].s,
            "input_zero_point": inputs[0].z,
            "output_scale": s,
            "ouput_zero_point": z,
            "output_dtype": torch.int8,
            "alpha": 0,
        })

    def replace_module(self, module):
        return _WrapperModule(lambda x: QLinearLeakyReluFn.apply(x, *self.args[module].values()))


class QuantizedAddHandler(Handler):
    def forward_hook(self, module, inputs, outputs):
        s, z = module.calc_output_scale_and_zero_point()
        self.args[module] = {
            "a_scale": torch.tensor(inputs[0].s.item()),
            "a_zero_point": torch.tensor(inputs[0].z.item(), dtype=torch.int8),
            "b_scale": torch.tensor(inputs[1].s.item()),
            "b_zero_point": torch.tensor(inputs[1].z.item(), dtype=torch.int8),
            "output_scale": torch.tensor(s.item()),
            "output_zero_point": torch.tensor(z.item(), dtype=torch.int8),
            "output_dtype": torch.int8,
        }

    def replace_module(self, module):
        return _WrapperModule(
            lambda x, y: QLinearAddFn.apply(
                x,
                self.args[module]["a_scale"],
                self.args[module]["a_zero_point"],
                y,
                self.args[module]["b_scale"],
                self.args[module]["b_zero_point"],
                self.args[module]["output_scale"],
                self.args[module]["output_zero_point"],
                self.args[module]["output_dtype"],
            ))


class QuantizedAdaptiveAvgPool2dHandler(Handler):
    def forward_hook(self, module, inputs, outputs):
        self.args[module] = OrderedDict({
            "input_scale": inputs[0].s,
            "input_zero_point": inputs[0].z,
            "output_scale": outputs.s,
            "output_zero_point": outputs.z,
            "output_dtype": torch.int8,
            "out_shape": tuple(outputs.q.shape),
            "channels_last": 0,
        })
        assert module.output_size == (1, 1)

    def replace_module(self, module):
        return _WrapperModule(lambda x: QLinearGlobalAveragePoolFn.apply(x, *self.args[module].values()))


class QuantizedMaxPool2dHandler(Handler):
    def forward_hook(self, module, inputs, outputs):
        kernel_shape = module.kernel_size
        pads = module.padding
        strides = module.stride
        if isinstance(kernel_shape, int):
            kernel_shape = (kernel_shape, kernel_shape)
        if isinstance(pads, int):
            pads = (pads, pads, pads, pads)
        if strides is None:
            strides = (1, 1)
        elif isinstance(strides, int):
            strides = (strides, strides)
        self.args[module] = OrderedDict({
            "out_shape": tuple(outputs.q.shape),
            "output_dtype": torch.int8,
            "kernel_shape": kernel_shape,
            "pads": pads,
            "strides": strides,
        })
        assert module.dilation == 1, module.dilation

    def replace_module(self, module):
        return _WrapperModule(lambda x: MaxPoolFn.apply(x, *self.args[module].values()))


class QuantizedLinearHandler(Handler):
    def forward_hook(self, module, inputs, outputs):
        quantized_weight = module._quantize_weight(module.linear.weight)
        quantized_bias = module._quantize_bias(
            inputs[0], quantized_weight, module.linear.bias)

        self.args[module] = OrderedDict({
            "a_scale": inputs[0].s.detach(),
            "a_zero_point": inputs[0].z,
            "int_b": quantized_weight.q,
            "b_scale": quantized_weight.s.detach(),
            "b_zero_point": quantized_weight.z,
            "int_c": quantized_bias.q,
            "output_scale": outputs.s.detach(),
            "output_zero_point": outputs.z,
            "output_dtype": torch.int8,
            "out_shape": tuple(outputs.q.shape),
            "trans_a": 0,
            "trans_b": 1,
            "alpha": 1,
        })

    def replace_module(self, module):
        return _WrapperModule(lambda x: QGemmFn.apply(x, *self.args[module].values()))


class QuantizedFlattenHandler(Handler):
    def forward_hook(self, module, inputs, outputs):
        self.args[module] = {
            "dequant": OrderedDict({
                "input_scale": inputs[0].s,
                "input_zero_point": inputs[0].z,
                "input_axis": None
            }),
            "flatten": {
                "start_dim": module.start_dim,
                "end_dim": module.end_dim,
            },
            "quant": OrderedDict({
                "output_scale": inputs[0].s,
                "output_zero_point": inputs[0].z,
                "output_dtype": torch.int8,
                "output_axis": None
            })
        }

    def replace_module(self, module):
        return nn.Sequential(
            _WrapperModule(
                lambda x: DequantizeLinearFn.apply(x, *self.args[module]["dequant"].values())),
            nn.Flatten(**self.args[module]["flatten"]),
            _WrapperModule(
                lambda x: QuantizeLinearFn.apply(x, *self.args[module]["quant"].values()))
        )
