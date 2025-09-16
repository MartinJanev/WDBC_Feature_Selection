import numpy as np
import random
from dataclasses import dataclass

SEED = 42

def set_seed(seed: int = SEED):
    np.random.seed(seed)
    random.seed(seed)

@dataclass
class CI:
    """
    Confidence Interval
    A simple dataclass to hold mean, 95% confidence interval bounds, standard deviation, and sample size.
    """
    mean: float
    lo95: float
    hi95: float
    std: float
    n: int

def mean_ci(values):
    """
    Compute mean and 95% confidence interval for a list or array of values.
    Uses the formula: CI = mean ± 1.96 * (std / sqrt(n))
    :param values: List or array of numerical values
    :return: CI dataclass instance with mean, lo95, hi95, std, and n
    """
    arr = np.asarray(values, dtype=float)
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    n = len(arr)
    err = 1.96 * (s / np.sqrt(n)) if n > 1 else 0.0
    return CI(mean=m, lo95=m - err, hi95=m + err, std=s, n=n)
