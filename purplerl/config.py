import os.path as osp

import torch

dtype = torch.float32

def use_gpu():
    global device, pin, tensor_args
    device = torch.device('cuda')

    pin = True

    tensor_args = {
        "dtype": dtype,
        "device": device
    }

def use_cpu():
    global device, pin, tensor_args
    device = torch.device('cpu')

    pin = False

    tensor_args = {
        "dtype": dtype,
        "device": device
    }

use_gpu()
