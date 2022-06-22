import os.path as osp

import torch

device = torch.device('cuda')

tensor_args = {
    "dtype": torch.float32, 
    "device": device
}