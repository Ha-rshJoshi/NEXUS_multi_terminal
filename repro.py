"""Seeds Python/NumPy/PyTorch randomness so model training is deterministic
across runs -- called by every entrypoint that builds or trains a model."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Binds Python, NumPy and PyTorch to a single seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (only matters if running on GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
