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
