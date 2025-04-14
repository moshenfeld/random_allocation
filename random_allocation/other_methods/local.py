from __future__ import annotations
from typing import Callable
from functools import cache
import numpy as np
from scipy import stats
from dp_accounting import pld

# ================= Supporting function =================
def bin_search(func: Callable, lower: float, upper: float, target: float, increasing: bool = False) -> float:
    search_params = pld.common.BinarySearchParameters(lower, upper, (upper - lower) / 1_000)
    if increasing:
        return pld.common.monotone_function(func, target, search_params)
    return pld.common.inverse_monotone_function(lambda val: func(val), target, search_params)

# ==================== Deterministic ====================
@cache
def deterministic_delta(sigma: float,
                        epsilon: float,
                        ) -> float:
    upper_cdfs = stats.norm.cdf(0.5 / sigma - sigma * epsilon)
    lower_log_cdfs = stats.norm.logcdf(-0.5 / sigma - sigma * epsilon)
    return upper_cdfs - np.exp(epsilon + lower_log_cdfs)

@cache
def deterministic_epsilon(sigma: float,
                          delta: float,
                          epsilon_upper_bound: float = 100,
                          ) -> float:
    epsilon = bin_search(lambda eps: deterministic_delta(sigma=sigma, epsilon=eps),
                         0, epsilon_upper_bound, delta, increasing=False)
    return np.inf if epsilon is None else epsilon

# ==================== Local ====================
@cache
def local_delta(sigma: float,
                epsilon: float,
                num_selected: int,
                num_epochs: int,
                ) -> np.ndarray[float]:
    return deterministic_delta(sigma=sigma/np.sqrt(num_selected*num_epochs), epsilon=epsilon)

@cache
def local_epsilon(sigma: float,
                  delta: float,
                  num_selected: int,
                  num_epochs: int,
                  epsilon_upper_bound: float = 100,
                  ) -> float:
    return deterministic_epsilon(sigma=sigma/np.sqrt(num_selected*num_epochs), delta=delta, epsilon_upper_bound=epsilon_upper_bound)