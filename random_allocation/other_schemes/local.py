from __future__ import annotations
import numpy as np
from scipy import stats

from random_allocation.comparisons.utils import search_function_with_bounds, FunctionType
from random_allocation.comparisons.structs import PrivacyParams, SchemeConfig

# ==================== Deterministic ====================
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

def Gaussian_epsilon(sigma: float,
                     delta: float,
                     epsilon_tolerance: float = 1e-3,
                     ) -> float:
    """
    Calculate the epsilon privacy parameter of the Gaussian mechanism for a given delta.

    Parameters:
    - sigma: The standard deviation of the Gaussian mechanism.
    - delta: The privacy profile bound.
    - tolerance: The acceptable error margin.

    Returns:
    - The calculated epsilon value or infinity if not found.
    """
    # Compute the analytic upper bound for epsilon
    epsilon_upper_bound = 1/(2*sigma**2) + np.sqrt(2*np.log(sigma/delta*np.sqrt(2/np.pi)))/sigma

    # Find the epsilon value using binary search
    optimization_func = lambda eps: Gaussian_delta(sigma=sigma, epsilon=eps)
    epsilon = search_function_with_bounds(func=optimization_func, y_target=delta, bounds=(0, epsilon_upper_bound),
                                          tolerance=epsilon_tolerance, function_type=FunctionType.DECREASING)
    return np.inf if epsilon is None else epsilon

# ==================== Local ====================
def local_delta(params: PrivacyParams,
                config: SchemeConfig = SchemeConfig(),
                ) -> float:
    """
    Calculate the privacy profile in case the index where each element is used is public (no amplification).

    Parameters:
    - params: PrivacyParams containing sigma, epsilon, num_selected, and num_epochs
    - config: Configuration parameters (not used for this calculation)
    """
    params.validate()
    if params.epsilon is None:
        raise ValueError("Epsilon must be provided to compute delta")
        
    return Gaussian_delta(sigma=params.sigma/np.sqrt(params.num_selected*params.num_epochs), 
                         epsilon=params.epsilon)

def local_epsilon(params: PrivacyParams,
                  config: SchemeConfig = SchemeConfig(),
                  ) -> float:
    """
    Calculate the local epsilon value based on sigma, delta, number of selections, and epochs.

    Parameters:
    - params: PrivacyParams containing sigma, delta, num_selected, and num_epochs
    - config: Configuration parameters
    """
    params.validate()
    if params.delta is None:
        raise ValueError("Delta must be provided to compute epsilon")
    
    return Gaussian_epsilon(sigma=params.sigma/np.sqrt(params.num_selected*params.num_epochs), 
                           delta=params.delta, 
                           epsilon_tolerance=config.epsilon_tolerance)