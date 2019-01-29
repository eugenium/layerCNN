'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.datasets.cifar import CIFAR10

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

from torch.nn.parallel import DataParallel
from torch.nn.parallel.scatter_gather import scatter_kwargs,scatter
class DataParallelSpecial(DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallelSpecial,self).__init__(module, device_ids=None, output_device=None, dim=0)
        print('Initialized with GPUs:')
        print(self.device_ids)

    def forward(self, *inputs, init=False, **kwargs):
        if init:
            if self.device_ids:
                # -------- Here, we split the input tensor across GPUs
                inputs_ = inputs
                if not isinstance(inputs_, tuple):
                    inputs_ = (inputs_,)

                representation, _ = scatter_kwargs(inputs_, None, self.device_ids, 0)
                self.replicas = self.replicate(self.module, self.device_ids[:len(representation)])
                # ----
            else:
                representation = inputs
            return None , representation

        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        # inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
        if len(self.device_ids) == 1:
            import ipdb; ipdb.set_trace()
            return self.module(*inputs[0][0], **kwargs)

        kwargs = scatter(kwargs, self.device_ids) if kwargs else []
        kwargs = tuple(kwargs)
        outputs = self.parallel_apply(self.replicas, *inputs, kwargs)

        out1 = []
        out2 = []
        for i, tensor in enumerate(outputs):
            with torch.cuda.device(tensor[0].get_device()):
                # out_1[i] = torch.autograd.Variable(tensors[i])
                out1.append(outputs[i][0])
                out2.append(outputs[i][1])
        outputs = self.gather(out1, self.output_device)
        representation = out2
        return outputs, representation

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def convnet_half_precision(model):
    model.half()  # convert to half precision
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
        if isinstance(layer, nn.BatchNorm1d):
            layer.float()
    return model

def reset(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        m.bias.data.zero_()
