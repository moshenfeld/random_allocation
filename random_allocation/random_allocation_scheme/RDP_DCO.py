# Standard library imports
from typing import Callable, List, Union, Optional, Tuple, Dict, Any, cast, Literal

# Third-party imports
import numpy as np
from numpy.typing import NDArray

# Local application imports
from random_allocation.random_allocation_scheme.direct import log_factorial_range, log_factorial
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig, Direction, Verbosity

# Type alias for numpy float arrays
FloatArray = NDArray[np.float64]

def allocation_RDP_DCO_remove(sigma: float,
                              num_steps: int,
                              num_selected: int,
                              alpha: float,
                              ) -> float:
    """
    Compute an upper bound on RDP of the allocation mechanism based on alpha=2
    
    Args:
        sigma: Noise scale
        num_steps: Number of steps 
        num_selected: Number of selected items
        alpha: Alpha order for RDP
    
    Returns:
        Upper bound on RDP
    """
    log_terms_arr = np.array([log_factorial_range(n=num_selected, m=i) - log_factorial(n=i)
                              + log_factorial_range(n=num_steps-num_selected, m=num_selected-i) - log_factorial(n=num_selected-i)
                              + i*alpha/(2*sigma**2) for i in range(num_selected+1)])
    max_log_term = np.max(log_terms_arr)
    return float(max_log_term + np.log(np.sum(np.exp(log_terms_arr - max_log_term))) - 
                 log_factorial_range(n=num_steps, m=num_selected) + log_factorial(n=num_selected))

def allocation_RDP_DCO_add(sigma: float,
                           num_steps: int,
                           num_selected: int,
                           alpha: float,
                           ) -> float:
    """
    Compute an upper bound on RDP of the allocation mechanism (add direction)
    
    Args:
        sigma: Noise scale
        num_steps: Number of steps
        num_selected: Number of selected items
        alpha: Alpha order for RDP
    
    Returns:
        Upper bound on RDP
    """
    return float(alpha*num_selected**2/(2*sigma**2*num_steps) \
        + (alpha*num_selected*(num_steps-num_selected))/(2*sigma**2*num_steps*(alpha-1)) \
        - num_steps*np.log1p(alpha*(np.exp(num_selected*(num_steps-num_selected)/(sigma**2*num_steps**2))-1))/(2*(alpha-1)))

# ==================== Both ====================
def allocation_epsilon_RDP_DCO(params: PrivacyParams, 
                               config: SchemeConfig,
                               direction: Direction = Direction.BOTH,
                               ) -> float:
    """
    Compute epsilon for the RDP-DCO allocation scheme.
    
    Args:
        params: Privacy parameters
        config: Scheme configuration parameters
        direction: The direction of privacy. Can be ADD, REMOVE, or BOTH.
    
    Returns:
        Computed epsilon value
    """
    params.validate()
    if params.delta is None:
        raise ValueError("Delta must be provided to compute epsilon")
    
    # Use alpha_orders directly from config or generate if not provided
    if config.allocation_RDP_DCO_alpha_orders is not None:
        alpha_orders: NDArray[np.float64] = np.array(config.allocation_RDP_DCO_alpha_orders, dtype=np.float64)
        assert np.all(alpha_orders >= 2), f"All alpha values must be >= 2 for RDP_DCO. Found min value: {np.min(alpha_orders)}"
    else:
        alpha_orders = np.concatenate((np.arange(2, 202), np.exp(np.linspace(np.log(202), np.log(10_000), 50))))
    
    # Variables that will be defined conditionally
    epsilon_add: float  # type annotation without initialization
    epsilon_remove: float  # type annotation without initialization
    used_alpha_add: float  # type annotation without initialization
    used_alpha_remove: float  # type annotation without initialization
    
    # Compute RDP and epsilon values
    if direction != Direction.ADD:
        alpha_RDP = params.num_epochs * np.array([allocation_RDP_DCO_remove(
            params.sigma, params.num_steps, params.num_selected, float(alpha))
            for alpha in alpha_orders])
        alpha_epsilons = alpha_RDP + np.log1p(-1/alpha_orders) - np.log(params.delta * alpha_orders)/(alpha_orders-1)
        epsilon_remove = float(np.min(alpha_epsilons))
        used_alpha_remove = float(alpha_orders[np.argmin(alpha_epsilons)])
    
    if direction != Direction.REMOVE:
        alpha_RDP = params.num_epochs * np.array([allocation_RDP_DCO_add(
            params.sigma, params.num_steps, params.num_selected, float(alpha))
            for alpha in alpha_orders])
        alpha_epsilons = alpha_RDP + np.log1p(-1/alpha_orders) - np.log(params.delta * alpha_orders)/(alpha_orders-1)
        epsilon_add = float(np.min(alpha_epsilons))
        used_alpha_add = float(alpha_orders[np.argmin(alpha_epsilons)])

    # Determine the epsilon and used alpha based on the direction
    if direction == Direction.ADD:
        assert 'epsilon_add' in locals() and 'used_alpha_add' in locals(), "Failed to compute epsilon_add"
        epsilon = epsilon_add
        used_alpha = used_alpha_add
    elif direction == Direction.REMOVE:
        assert 'epsilon_remove' in locals() and 'used_alpha_remove' in locals(), "Failed to compute epsilon_remove"
        epsilon = epsilon_remove
        used_alpha = used_alpha_remove
    else:
        assert 'epsilon_add' in locals() and 'epsilon_remove' in locals(), "Failed to compute either epsilon_add or epsilon_remove"
        epsilon = max(epsilon_add, epsilon_remove)
        used_alpha = used_alpha_add if epsilon_add > epsilon_remove else used_alpha_remove

    # Check for potential alpha overflow or underflow
    if config.verbosity != Verbosity.NONE:
        if used_alpha == alpha_orders[-1]:
            print(f'Potential alpha overflow! used alpha: {used_alpha} which is the maximal alpha')
        if used_alpha == alpha_orders[0]:
            print(f'Potential alpha underflow! used alpha: {used_alpha} which is the minimal alpha')

    # Print debug info if requested
    if config.verbosity == Verbosity.ALL:
        print(f'sigma: {params.sigma}, delta: {params.delta}, num_steps: {params.num_steps}, '
              f'num_selected: {params.num_selected}, num_epochs: {params.num_epochs}, used_alpha: {used_alpha}')
    
    return epsilon

def allocation_delta_RDP_DCO(params: PrivacyParams,
                             config: SchemeConfig,
                             direction: Direction = Direction.BOTH,
                             ) -> float:
    """
    Compute delta for the RDP-DCO allocation scheme.
    
    Args:
        params: Privacy parameters
        config: Scheme configuration parameters
        direction: The direction of privacy. Can be ADD, REMOVE, or BOTH.
        
    Returns:
        Computed delta value
    """
    params.validate()
    if params.epsilon is None:
        raise ValueError("Epsilon must be provided to compute delta")
    
    # Use alpha_orders directly from config or generate if not provided
    if config.allocation_RDP_DCO_alpha_orders is not None:
        alpha_orders: NDArray[np.float64] = np.array(config.allocation_RDP_DCO_alpha_orders, dtype=np.float64)
        assert np.all(alpha_orders >= 2), f"All alpha values must be >= 2 for RDP_DCO. Found min value: {np.min(alpha_orders)}"
    else:
        alpha_orders = np.concatenate((np.arange(2, 202), np.exp(np.linspace(np.log(202), np.log(10_000), 50))))
    
    # Variables that will be defined conditionally
    delta_add: float  # type annotation without initialization
    delta_remove: float  # type annotation without initialization
    used_alpha_add: float  # type annotation without initialization
    used_alpha_remove: float  # type annotation without initialization
    
    # Compute RDP and delta values using log-space calculations
    if direction != Direction.ADD:
        # Compute RDP values
        alpha_RDP = params.num_epochs * np.array([allocation_RDP_DCO_remove(
            params.sigma, params.num_steps, params.num_selected, float(alpha))
            for alpha in alpha_orders])
        
        # Compute log(delta) directly to avoid overflow
        log_alpha_deltas = (alpha_orders-1) * (alpha_RDP - params.epsilon) + \
                         alpha_orders * np.log1p(-1/alpha_orders) - np.log(alpha_orders-1)
        
        # Find the minimum delta and corresponding alpha directly in log space
        min_log_delta_idx = np.argmin(log_alpha_deltas)
        delta_remove = float(np.exp(log_alpha_deltas[min_log_delta_idx]))
        used_alpha_remove = float(alpha_orders[min_log_delta_idx])
    
    if direction != Direction.REMOVE:
        # Compute RDP values
        alpha_RDP = params.num_epochs * np.array([allocation_RDP_DCO_add(
            params.sigma, params.num_steps, params.num_selected, float(alpha))
            for alpha in alpha_orders])
        
        # Compute log(delta) directly to avoid overflow
        log_alpha_deltas = (alpha_orders-1) * (alpha_RDP - params.epsilon) + \
                         alpha_orders * np.log1p(-1/alpha_orders) - np.log(alpha_orders-1)
        
        # Find the minimum delta and corresponding alpha directly in log space
        min_log_delta_idx = np.argmin(log_alpha_deltas)
        delta_add = float(np.exp(log_alpha_deltas[min_log_delta_idx]))
        used_alpha_add = float(alpha_orders[min_log_delta_idx])

    # Determine the delta and used alpha based on the direction
    if direction == Direction.ADD:
        assert 'delta_add' in locals() and 'used_alpha_add' in locals(), "Failed to compute delta_add"
        delta = delta_add
        used_alpha = used_alpha_add
    elif direction == Direction.REMOVE:
        assert 'delta_remove' in locals() and 'used_alpha_remove' in locals(), "Failed to compute delta_remove"
        delta = delta_remove
        used_alpha = used_alpha_remove
    else:
        assert 'delta_add' in locals() and 'delta_remove' in locals(), "Failed to compute either delta_add or delta_remove"
        delta = max(delta_add, delta_remove)
        used_alpha = used_alpha_add if delta_add > delta_remove else used_alpha_remove

    # Check for potential alpha overflow or underflow
    if config.verbosity != Verbosity.NONE:
        if used_alpha == alpha_orders[-1]:
            print(f'Potential alpha overflow! used alpha: {used_alpha} which is the maximal alpha')
        if used_alpha == alpha_orders[0]:
            print(f'Potential alpha underflow! used alpha: {used_alpha} which is the minimal alpha')

    # Print debug info if requested
    if config.verbosity == Verbosity.ALL:
        print(f'sigma: {params.sigma}, epsilon: {params.epsilon}, num_steps: {params.num_steps}, '
              f'num_selected: {params.num_selected}, num_epochs: {params.num_epochs}, used_alpha: {used_alpha}')
    
    return delta