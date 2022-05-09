# Quantization Aware Training

This repository provides a PyTorch re-implementation of the quantization-aware training (QAT) algorithm,
which is firstly introduced by the paper: 
[Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877).
The core insight of the algorithm is to simulate the quantization effect in the forward pass,
but backpropagate gradients in the original float computational graph.
So the computational graph can both be aware of the precision loss caused by quantization
and nudge the float weights at the same time.
Using different logic for forward and backward pass is done by the PyTorch `detach` operator.
For example, `a - (a - b).detach()` will yield `b` in the forward pass.
But the gradients will directly be passed to `a` and ignore `(a - b).detach()` in the backward pass.

## Results

The following are the quantization aware training results of ResNet-18 on ImageNet dataset.
Floating point accuracy is directly copied from PyTorch official site.
The QAT version of ResNet-18 is trained for 100 epochs from scratch with network structure defined in [qat/networks/resnet.py](qat/networks/resnet.py).
The learning rate starts from 0.1 and decays by a factor of 0.1 every 30 epochs.
We can see that the results basically reflect the ability of QAT.

|ResNet-18 (F32)|ResNet-18 (INT8 QAT)|
|-|-|
|69.76%|68.82%|

## Export to ONNX

This repository supports exporting PyTorch training graph to ONNX format with one line code.
This is completed by bridging PyTorch quantization module and ONNX operator with symbolic `autograd.Function`.
The following is the one line code to export a training module to its int8 ONNX inference format.

```python
from qat.export.export import export
export(model, (torch.randn(1, 3, 224, 224),), "model.quant.onnx")
```
