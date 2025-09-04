import numpy as np
import random
from dataclasses import dataclass

SEED = 42

def set_seed(seed: int = SEED):
    np.random.seed(seed)
    random.seed(seed)

@dataclass
class CI:
    mean: float
    lo95: float
    hi95: float
    std: float
    n: int

def mean_ci(values):
    arr = np.asarray(values, dtype=float)
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    n = len(arr)
    err = 1.96 * (s / np.sqrt(n)) if n > 1 else 0.0
    return CI(mean=m, lo95=m - err, hi95=m + err, std=s, n=n)
