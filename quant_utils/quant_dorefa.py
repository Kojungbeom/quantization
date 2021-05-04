#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zzzxxxttt
    Edited by Kojungbeom
    
https://github.com/zzzxxxttt/pytorch_DoReFaNet/blob/master/utils/quant_dorefa.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def uniform_quantize(k):
    class qfn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2 ** k - 1)
                out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input

    return qfn().apply


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        assert w_bit <= 8 or w_bit == 32
        self.w_bit = w_bit
        self.uniform_q = uniform_quantize(k=w_bit)

    def forward(self, x):
        if self.w_bit == 32:
            weight_q = x
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(x)).detach()
            weight_q = self.uniform_q(x / E) * E
        else:
            #map the weights between [-0.5, 0.5]
            weight = torch.tanh(x)
            max_w = torch.max(torch.abs(weight)).detach()
            weight = weight / 2 / max_w + 0.5
            weight_q = 2 * self.uniform_q(weight) - 1
        return weight_q


def conv2d_dorefa(w_bit):
    class Conv2d_dorefa(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, bias=False):
            super(Conv2d_dorefa, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
            self.w_bit = w_bit
            self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

        def forward(self, input, order=None):
            input = input
            weight_q = self.quantize_fn(self.weight)
            return F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

    return Conv2d_dorefa


def linear_dorefa(w_bit):
    class Linear_dorefa(nn.Linear):
        def __init__(self, in_features, out_features, bias=True):
            super(Linear_dorefa, self).__init__(in_features, out_features, bias)
            self.w_bit = w_bit
            self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

        def forward(self, input):
            weight_q = self.quantize_fn(self.weight)
            return F.linear(input, weight_q, self.bias)

    return Linear_dorefa

