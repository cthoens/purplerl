import os.path as osp

import torch

use_gpu = False

if use_gpu:
    device = torch.device('cuda')

    pin = True

    tensor_args = {
        "dtype": torch.float32, 
        "device": device
    }
else:
    device = torch.device('cpu')

    pin = False

    tensor_args = {
        "dtype": torch.float32, 
        "device": device
    }