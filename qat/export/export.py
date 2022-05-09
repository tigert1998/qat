from copy import deepcopy

import torch
import torch.nn as nn

from qat.export.handlers import *
from qat.ops import *
from .utils import *


def export(model: nn.Module, args, f):
    copyed_model = deepcopy(model)

    hooks = []
    handlers = {
        Quantize: QuantizeHandler(),
        QuantizedConv2dBatchNorm2dReLU: QuantizedConv2dBatchNorm2dReLUHandler(),
        QuantizedReLU: QuantizedReLUHandler(),
        QuantizedAdd: QuantizedAddHandler(),
        QuantizedAdaptiveAvgPool2d: QuantizedAdaptiveAvgPool2dHandler(),
        QuantizedMaxPool2d: QuantizedMaxPool2dHandler(),
        QuantizedLinear: QuantizedLinearHandler(),
        QuantizedFlatten: QuantizedFlattenHandler(),
    }

    def _hook(module, inputs, outputs):
        for key in handlers:
            if isinstance(module, key):
                handlers[key].forward_hook(module, inputs, outputs)

    for module in copyed_model.modules():
        hooks.append(module.register_forward_hook(_hook))

    copyed_model.eval()
    copyed_model(*args)

    for hook in hooks:
        hook.remove()

    new_modules = {}

    for name, module in copyed_model.named_modules():
        if type(module) not in handlers:
            continue
        handler = handlers[type(module)]
        new_modules[name] = handler.replace_module(module)

    for name, new_module in new_modules.items():
        replace_module_by_name(copyed_model, name, new_module)

    torch.onnx.export(copyed_model, args, f, opset_version=OPSET)
