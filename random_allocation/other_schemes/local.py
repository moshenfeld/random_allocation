from __future__ import annotations
from typing import Callable
from functools import cache
import numpy as np
from scipy import stats
from dp_accounting import pld

# ================= Supporting function =================
def bin_search(func: Callable, lower: float, upper: float, target: float, tolerance: float, increasing: bool) -> float:
    """
    Perform binary search on a monotonic function.

    Parameters:
    - func: The function to search.
    - lower: The lower bound of the search.
    - upper: The upper bound of the search.
    - target: The target value to find.
    - tolerance: The acceptable error margin.
    - increasing: Boolean indicating if the function is increasing.
    """
    search_params = pld.common.BinarySearchParameters(lower_bound=lower, upper_bound=upper, tolerance=tolerance)
    return pld.common.inverse_monotone_function(func, target, search_params, increasing)

# ==================== Deterministic ====================
# @cache
def Gaussian_delta(sigma: float,
                   epsilon: float,
                   ) -> float:
    """
    Calculate the privacy profile of the Gaussian mechanism for a given epsilon.

    Parameters:
    - sigma: The standard deviation of the Gaussian mechanism.
    - epsilon: The privacy parameter.
    """
    upper_cdfs = stats.norm.cdf(0.5 / sigma - sigma * epsilon)
    lower_log_cdfs = stats.norm.logcdf(-0.5 / sigma - sigma * epsilon)
    return upper_cdfs - np.exp(epsilon + lower_log_cdfs)

# @cache
def Gaussian_epsilon(sigma: float,
                     delta: float,
                     tolerance: float = 0.001,
                     epsilon_upper_bound: float = 100,
                     ) -> float:
    """
    Calculate the epsilon privacy parameter of the Gaussian mechanism for a given delta.

    Parameters:
    - sigma: The standard deviation of the Gaussian mechanism.
    - delta: The privacy profile bound.
    - tolerance: The acceptable error margin.
    - epsilon_upper_bound: The upper bound for epsilon.

    Returns:
    - The calculated epsilon value or infinity if not found.
    """
    epsilon = bin_search(lambda eps: Gaussian_delta(sigma=sigma, epsilon=eps),
                         lower=0, upper=epsilon_upper_bound, target=delta, tolerance=tolerance, increasing=False)
    return np.inf if epsilon is None else epsilon

# ==================== Local ====================
# @cache
def local_delta(sigma: float,
                epsilon: float,
                num_selected: int,
                num_epochs: int,
                ) -> np.ndarray[float]:
    """
    Calculate the privacy profile in case the index where each element is used is public (no amplification).

    Parameters:
    - sigma: The standard deviation of the Gaussian mechanism.
    - epsilon: The privacy parameter.
    - num_selected: The number of steps that an element is used per epoch
    - num_epochs: The number of epochs.
    """
    return Gaussian_delta(sigma=sigma/np.sqrt(num_selected*num_epochs), epsilon=epsilon)

# @cache
def local_epsilon(sigma: float,
                  delta: float,
                  num_selected: int,
                  num_epochs: int,
                  ) -> float:
    """
    Calculate the local epsilon value based on sigma, delta, number of selections, and epochs.

    Parameters:
    - sigma: The standard deviation of the Gaussian mechanism.
    - delta: The privacy profile bound.
    - num_selected: The number of steps that each element is used per epoch
    - num_epochs: The number of epochs.
    """
    return Gaussian_epsilon(sigma=sigma/np.sqrt(num_selected*num_epochs), delta=delta)