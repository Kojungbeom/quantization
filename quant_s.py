#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 21:55:12 2020

@author: jisu
"""

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

'''
def conv2d_Q_fn(w_bit):
    class Conv2d_Q(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, bias=False, sigma=0.4):
            super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                  padding, dilation, groups, bias)
            self.sigma = sigma
            self.threshold = 0
            self.w_bit = w_bit
            self.quantize_fn = weight_quantize_fn(w_bit=w_bit, threshold=self.threshold)
            self.weight_shape = self.weight.shape
            
        def sparse_weight(self, sigma):
            sigma = sigma
            threshold = torch.mean(self.weight) + torch.std(self.weight) * sigma
            self.threshold = threshold.cuda()
            weight_shape = self.weight.shape
            mask = torch.full((self.weight_shape[0], self.weight_shape[1],
                               self.weight_shape[2], self.weight_shape[3]), float(self.threshold))
            mask = mask.cuda()
            mask = torch.where(mask > torch.abs(self.weight), 0, 1)
            prune.custom_from_mask(self, name='weight', mask=mask)
        
        def forward(self, input, order=None):
            input = input
            weight_q = self.sparse_weight(sigma=self.sigma)
            weight_sq = self.quantize_fn(self.weight)
            #weight_q = torch.cuda.FloatTensor(weight_q)
            # print(np.unique(weight_q.detach().numpy()))
            return F.conv2d(input, weight_sq, self.bias, self.stride, self.padding, self.dilation, self.groups)

    return Conv2d_Q

def sparse_weight(self, sigma):
    sigma = sigma
    threshold = torch.mean(self.weight) + torch.std(self.weight) * sigma
    self.threshold = threshold.cuda()
    weight_shape = self.weight.shape
    mask = torch.full((weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]), float(self.threshold))
    mask = mask.cuda()
    mask = torch.where(mask > torch.abs(self.weight), 0, 1)
    prune.custom_from_mask(self, name='weight', mask=mask)


def minmax_uniform_quantize(k, threshold):
    class qfn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2 ** k - 1)
                w_max = torch.max(input)
                w_min = threshold
                w_s = torch.div(torch.abs(input) - w_min, w_max - w_min)                
                w_q = torch.round(w_s * n) / n
                w_sq = torch.mul(torch.sign(input), torch.mul(w_q, w_max-w_min)+w_min) 
            return w_sq
    
        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input
    
    return qfn().apply

class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit, threshold):
        super(weight_quantize_fn, self).__init__()
        assert w_bit <= 8 or w_bit == 32
        self.w_bit = w_bit
        self.uniform_q = minmax_uniform_quantize(k=w_bit, threshold=threshold)
        self.threshold = threshold
    def forward(self, x):
        if self.w_bit == 32:
            weight_sq = x
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(x)).detach()
            weight_sq = self.uniform_q(x / E) * E
        else:
            weight_sq = self.uniform_q(x)
          
        return weight_sq



def linear_Q_fn(w_bit):
    class Linear_Q(nn.Linear):
        def __init__(self, in_features, out_features, bias=True):
            super(Linear_Q, self).__init__(in_features, out_features, bias)
            self.w_bit = w_bit
            self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

        def forward(self, input):
            weight_q = self.quantize_fn(self.weight)
            # print(np.unique(weight_q.detach().numpy()))
            return F.linear(input, weight_q, self.bias)

    return Linear_Q
'''

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    a = torch.rand(1, 3, 32, 32)

    Conv2d = conv2d_Q_fn(w_bit=1)
    conv = Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

    img = torch.randn(1, 256, 56, 56)
    print(img.max().item(), img.min().item())
    out = conv(img)
    print(out.max().item(), out.min().item())
