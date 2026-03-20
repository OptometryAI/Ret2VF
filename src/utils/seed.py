import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True, warn_only: bool = False) -> None:
    """
    Set random seed for python, numpy, and torch.

    Args:
        seed: Random seed.
        deterministic: Whether to enable deterministic behavior for cudnn.
                       If True, training is more reproducible but may be slower.
                       If False, training may be faster but less reproducible.
        warn_only: When deterministic is True, whether to warn instead of raising
                   on operations without deterministic implementations.
    """
    if seed is None:
        return

    # Python built-in random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch CPU / GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # cuDNN settings
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
