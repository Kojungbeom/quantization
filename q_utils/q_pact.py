#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ychzhang
    Edited by Kojungbeom
    
https://github.com/cornell-zhang/dnn-gating
"""

import torch
import torch.nn as nn


class PactClip(torch.autograd.Function):
    """ Autograd function for PACTReLU
    """
    @staticmethod
    def forward(ctx, input, upper_bound, scale):
        """ upper_bound   if input > upper_bound
        y = input         if 0 <= input <= upper_bound
            0             if input < 0
        """
        ctx.save_for_backward(input, upper_bound)
        input = torch.clamp(input, 0, upper_bound.data)
        input = torch.round(input.mul_(scale/upper_bound))
        input = input.mul_(upper_bound/scale)
        #print(len(torch.unique(input)))
        return input

    @staticmethod
    def backward(ctx, grad_output):

        input, upper_bound, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_upper_bound = grad_output.clone()
        grad_input[input<0] = 0
        grad_input[input>upper_bound] = 0
        # Gradient for parameterized clipping level α
        grad_upper_bound[input<upper_bound] = 0

        return grad_input, torch.sum(grad_upper_bound), None


class PactReLU(nn.Module):
    """ PACTReLU
    """
    def __init__(self, upper_bound=10.0, a_bit=4):
        super(PactReLU, self).__init__()
        # Initial value is 10 at PACT paper
        self.upper_bound = nn.Parameter(torch.tensor(upper_bound))
        self.scale = pow(2, a_bit)-1

    def forward(self, input):
        return PactClip.apply(input, self.upper_bound, self.scale)
