#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.prune as prune


class squantization(torch.autograd.Function):
    # Min Max quantization for sparse weights
    @staticmethod
    def forward(ctx, input, scale, sigma):
        ctx.save_for_backward(input)

        abs_input = torch.abs(input)
        mean = torch.mean(abs_input)
        std = torch.std(abs_input)
        threshold = mean + std * sigma
        #f_threshold = float(threshold)
        w_mask = torch.full((input.shape[0], input.shape[1],
                             input.shape[2], input.shape[3]), threshold.detach()).cuda()
        w_mask = torch.where(w_mask > abs_input, 0, 1)
        w_sparse = input * w_mask

        w_max = torch.max(torch.abs(w_sparse))
        w_min = threshold
        
        w_norm_sparse = (torch.abs(w_sparse) - w_min) / (w_max - w_min)
        w_q = torch.round(scale * w_norm_sparse) / scale
        w_sq = torch.sign(w_sparse) * (w_q * (w_max-w_min) + w_min)
        w_sq = torch.where(w_sparse==0, w_sparse, w_sq)

        #print(len(torch.unique(w_sq)))
        #print(torch.unique(w_sq))
        return w_sq

    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradient only if weight != 0
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input == 0] = 0
        return grad_input, None, None


def conv2d_Q_fn(w_bit, sigma, delay):
    class Conv2d_Q(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, bias=False):
            super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                  padding, dilation, groups, bias)
            self.sigma = sigma
            self.scale = pow(2, (w_bit-1)) - 1	# 2^(k-1) - 1
            self.iter = 0
            self.w_bit = w_bit

        def forward(self, input, order=None):
            input = input
            
            if self.iter < 236 * delay :
                self.iter += 1
                return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            else :
                # use absolute value
                weight_sq = squantization.apply(self.weight, self.scale, self.sigma)
                self.iter += 1
                # self.weight should be full precision weight
                return F.conv2d(input, weight_sq, self.bias, self.stride, self.padding, self.dilation, self.groups)


    return Conv2d_Q
     
def linear_Q_fn(w_bit):
    class Linear_Q(nn.Linear):
        def __init__(self, in_features, out_features, bias=True):
            super(Linear_Q, self).__init__(in_features, out_features, bias)
            self.w_bit = w_bit
            self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

        def forward(self, input):
            weight_q = self.quantize_fn(self.weight)
            return F.linear(input, weight_q, self.bias)

    return Linear_Q
