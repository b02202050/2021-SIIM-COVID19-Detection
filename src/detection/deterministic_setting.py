""" Every randomness-related process should import this module
to provide the maximum degree of determinism. Note that some
CUDA operations will still non-deterministic w.r.t. these settings.
Please refer to https://pytorch.org/docs/stable/notes/randomness.html"""

import os
import random as rn
import numpy as np

import torch


def set_deterministic(seed=None):
    os.environ['PYTHONHASHSEED'] = str(seed if seed is not None else 100)
    rn.seed(seed if seed is not None else 12345)
    np.random.seed(seed if seed is not None else 101)
    torch.manual_seed(seed if seed is not None else 100)
    torch.cuda.manual_seed_all(seed if seed is not None else 102)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


set_deterministic()
