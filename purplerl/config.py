import os.path as osp

import torch

dtype = torch.float32

class GpuConfig:

    def __init__(self) -> None:
        self.device = torch.device('cuda')
        self.pin = True
        self.tensor_args = {
            "dtype": dtype,
            "device": self.device
        }

class CpuConfig():

    def __init__(self) -> None:
        self.device = torch.device('cpu')
        self.pin = False
        self.tensor_args = {
            "dtype": dtype,
            "device": self.device
        }


