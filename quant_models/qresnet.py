"""
@author: weiaicunzai
    Edited by Kojungbeom
    
https://github.com/weiaicunzai/pytorch-cifar100/
"""

import torch
import torch.nn as nn

from quant_utils.quant_sparse import *
from quant_utils.quant_pact import *
from quant_utils.quant_dorefa import *

quant_dict = {1:'d+p', 2:'sq', 3:'nosq'}

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """
    expansion = 1

    def __init__(self, wbit, abit, sigma, delay, qt, batch_size, in_channels, out_channels, stride=1):
        super().__init__()
        if qt == quant_dict[1]:
            Conv2d = conv2d_dorefa(w_bit=wbit)
        elif qt == quant_dict[2]:
            Conv2d = conv2d_SQ(w_bit=wbit, sigma=sigma, delay=delay, batch_size=batch_size)
        else:
            Conv2d = conv2d_Nosq(w_bit=wbit, delay=delay, batch_size=batch_size)

        #residual function
        self.residual_function = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            PactReLU(a_bit=abit),
            Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        self.activation_layer = nn.Sequential(PactReLU(a_bit=abit))
        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        output = self.residual_function(x) + self.shortcut(x)
        output = self.activation_layer(output)
        return output


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, wbit, abit, sigma, delay, qt, batch_size, in_channels, out_channels, stride=1):
        super().__init__()
        if qt == quant_dict[1]:
            Conv2d = conv2d_dorefa(w_bit=wbit)
        elif qt == quant_dict[2]:
            Conv2d = conv2d_SQ(w_bit=wbit, sigma=sigma, delay=delay, batch_size=batch_size)
        else:
            Conv2d = conv2d_Nosq(w_bit=wbit, delay=delay, batch_size=batch_size)

        self.residual_function = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            PactReLU(a_bit=abit),
            Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            PactReLU(a_bit=abit),
            Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion)
        )
        self.activation_layer = nn.Sequential(PactReLU(a_bit=abit))
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        output = self.residual_function(x) + self.shortcut(x)
        output = self.activation_layer(output)
        return output

class QResNet(nn.Module):
    def __init__(self,  block, num_block, batch_size=256, qt=None, wbit=4, abit=4, sigma=0, delay=70, num_classes=100):
        super().__init__()

        self.in_channels = 64
        #First and last layer exclude quantization
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, wbit=wbit, abit=abit, sigma=sigma, delay=delay, qt=qt, batch_size=batch_size)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, wbit=wbit, abit=abit, sigma=sigma, delay=delay, qt=qt, batch_size=batch_size)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, wbit=wbit, abit=abit, sigma=sigma, delay=delay, qt=qt, batch_size=batch_size)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, wbit=wbit, abit=abit, sigma=sigma, delay=delay, qt=qt, batch_size=batch_size)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, wbit, abit, sigma, delay, qt, batch_size):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(wbit, abit, sigma, delay, qt, batch_size, self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


def qresnet18(wbit, abit):
    return QResNet(BasicBlock, [2, 2, 2, 2], wbit=wbit, abit=abit, qt='d+p')

def qresnet34(wbit, abit):
    return QResNet(BasicBlock, [3, 4, 6, 3], wbit=wbit, abit=abit, qt='d+p')

def qresnet50(wbit, abit):
    return QResNet(BottleNeck, [3, 4, 6, 3], wbit=wbit, abit=abit, qt='d+p')


def sqresnet18(wbit, abit, sigma, delay, batch_size):
    return QResNet(BasicBlock, [2, 2, 2, 2], wbit=wbit, abit=abit, sigma=sigma, delay=delay, qt='sq', batch_size=batch_size)

def sqresnet34(wbit, abit, sigma, delay, batch_size):
    return QResNet(BasicBlock, [3, 4, 6, 3], wbit=wbit, abit=abit, sigma=sigma, delay=delay, qt='sq', batch_size=batch_size)

def sqresnet50(wbit, abit, sigma, delay, batch_size):
    return QResNet(BottleNeck, [3, 4, 6, 3], wbit=wbit, abit=abit, sigma=sigma, delay=delay, qt='sq', batch_size=batch_size)


def nos_qresnet18(wbit, abit, delay, batch_size):
    return QResNet(BasicBlock, [2, 2, 2, 2], wbit=wbit, abit=abit, delay=delay, qt='nosq', batch_size=batch_size)

def nos_qresnet34(wbit, abit, delay, batch_size):
    return QResNet(BasicBlock, [3, 4, 6, 3], wbit=wbit, abit=abit, delay=delay, qt='nosq', batch_size=batch_size)

def nos_qresnet50(wbit, abit, delay, batch_size):
    return QResNet(BottleNeck, [3, 4, 6, 3], wbit=wbit, abit=abit, delay=delay, qt='nosq', batch_size=batch_size)


'''
def s_noqresnet18(wbit, abit, sigma, delay):
    return QResNet(BasicBlock, [2, 2, 2, 2], sigma, delay, qt='snoq')

def s_noqresnet34(wbit, abit, sigma, delay):
    return QResNet(BasicBlock, [3, 4, 6, 3], sigma, delay, qt='snoq')

def s_noqresnet50(wbit, abit, sigma, delay):
    return QResNet(BottleNeck, [3, 4, 6, 3], sigma, delay, qt='snoq')
'''


