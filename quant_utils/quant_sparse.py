#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Feb 26 19:53:12 2021

@author: Kojungbeom
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.prune as prune

DATA_LEN = 60000

def minmax_uniform_quantize(k):
  class mmuq(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, threshold):
      ctx.save_for_backward(input, threshold)
      
      # Minmax uniform quantization
      n = float(2 ** (k-1) - 1)
      w_abs = torch.abs(input).detach()
      w_max = torch.max(w_abs).detach()
      w_min = threshold.detach()
      w_s = (w_abs - w_min) / (w_max - w_min)
      out = torch.round(w_s * n) / n
      out = torch.sign(input) * (out*(w_max - w_min) + w_min)
      return out

    @staticmethod
    def backward(ctx, grad_output):
      input, threshold, = ctx.saved_tensors
      grad_input = grad_output.clone()
      
      # mask out gradient with same mask
      grad_input[torch.abs(input) < threshold] = 0
      return grad_input, None

  return mmuq().apply


class weight_squantize(nn.Module):
    def __init__(self, w_bit, sigma):
        super(weight_squantize, self).__init__()
        assert w_bit <= 8 or w_bit == 32
        self.w_bit = w_bit
        self.sigma = sigma
        self.uniform_q = minmax_uniform_quantize(k=w_bit)

    def forward(self, x):
        # Calculate threshold
        w_abs = torch.abs(x)
        threshold = torch.mean(w_abs) + torch.std(w_abs) * self.sigma
        
        # Generate mask
        w_mask = torch.full((x.shape[0], x.shape[1],
                             x.shape[2], x.shape[3]), threshold.detach()).cuda()
        w_mask = torch.where(w_mask > w_abs, 0, 1)
        weight = x * w_mask
        w_sq = self.uniform_q(weight, threshold)
        return w_sq

class weight_nosquantize(nn.Module):
    def __init__(self, w_bit, sigma):
        super(weight_nosquantize, self).__init__()
        assert w_bit <= 8 or w_bit == 32
        self.w_bit = w_bit
        self.sigma = sigma
        self.uniform_q = minmax_uniform_quantize(k=w_bit)

    def forward(self, x):
        w_sq = self.uniform_q(weight, threshold)
        return w_sq



def conv2d_SQ(w_bit, sigma, delay):
    class Conv2d_SQ(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, bias=False):
            super(Conv2d_SQ, self).__init__(in_channels, out_channels, kernel_size, stride,
                  padding, dilation, groups, bias)
            self.sigma = sigma
            self.scale = pow(2, (w_bit-1)) - 1
            self.w_bit = w_bit
            self.quantize_fn = weight_squantize(w_bit=w_bit, sigma=sigma)
            self.delay =  delay
            self.iter = 0

        def forward(self, input, order=None):
            input = input
            # Delay algorithm
            if self.iter < (DATA_LEN / batch_size * delay) :
                self.iter += 1
                return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                weight_sq = self.quantize_fn(self.weight)
                self.iter += 1
                return F.conv2d(input, weight_sq, self.bias, self.stride, self.padding, self.dilation, self.groups)
            
    return Conv2d_SQ

def conv2d_Nosq(w_bit, delay):
    class Conv2d_Nosq(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, bias=False):
            super(Conv2d_Nosq, self).__init__(in_channels, out_channels, kernel_size, stride,
                  padding, dilation, groups, bias)
            self.scale = pow(2, (w_bit-1)) - 1
            self.w_bit = w_bit
            self.quantize_fn = weight_squantize(w_bit=w_bit, sigma=sigma)
            self.delay =  delay
            self.iter = 0

        def forward(self, input, order=None):
            input = input
            # Delay algorithm
            if self.iter < 236 * delay :
                self.iter += 1
                return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                weight_sq = self.quantize_fn(self.weight)
                self.iter += 1
                return F.conv2d(input, weight_sq, self.bias, self.stride, self.padding, self.dilation, self.groups)
            
    return Conv2d_SQ

def conv2d_Snoq(sigma, delay):
    class Conv2d_Snoq(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, bias=False):
            super(Conv2d_Snoq, self).__init__(in_channels, out_channels, kernel_size, stride,
                  padding, dilation, groups, bias)
            self.sigma = sigma
            self.scale = pow(2, (w_bit-1)) - 1
            self.w_bit = w_bit
            self.quantize_fn = weight_squantize(w_bit=w_bit, sigma=sigma)
            self.delay =  delay
            self.iter = 0

        def forward(self, input, order=None):
            input = input
            # Delay algorithm
            if self.iter < 236 * delay :
                self.iter += 1
                return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                weight_sq = self.quantize_fn(self.weight)
                self.iter += 1
                return F.conv2d(input, weight_sq, self.bias, self.stride, self.padding, self.dilation, self.groups)
            
    return Conv2d_SQ

