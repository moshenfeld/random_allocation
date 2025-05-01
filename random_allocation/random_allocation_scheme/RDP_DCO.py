import numpy as np
from typing import Callable

from random_allocation.random_allocation_scheme.direct import log_factorial_range, log_factorial
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig

def allocation_RDP_DCO_remove(sigma: float,
                              num_steps: int,
                              num_selected: int,
                              alpha: float,
                              ) -> float:
    ''' Compute an upper bound on RDP of the allocation mechanism based on alpha=2 '''
    log_terms_arr = np.array([log_factorial_range(n=num_selected, m=i) - log_factorial(n=i)
                              + log_factorial_range(n=num_steps-num_selected, m=num_selected-i) - log_factorial(n=num_selected-i)
                              + i*alpha/(2*sigma**2) for i in range(num_selected+1)])
    max_log_term = np.max(log_terms_arr)
    return max_log_term + np.log(np.sum(np.exp(log_terms_arr - max_log_term))) - log_factorial_range(n=num_steps, m=num_selected) + log_factorial(n=num_selected)

def allocation_RDP_DCO_add(sigma: float,
                           num_steps: int,
                           num_selected: int,
                           alpha: float,
                           ) -> float:
    return alpha*num_selected**2/(2*sigma**2*num_steps) + (alpha*num_selected*(num_steps-num_selected)/(sigma**2*num_steps) - num_steps*np.log(1 + alpha*(np.exp(num_selected*(num_steps-num_selected)/(sigma**2*num_steps**2))-1))) / (2*(alpha-1))

# ==================== Both ====================
def allocation_epsilon_RDP_DCO(params: PrivacyParams, 
                               config: SchemeConfig = SchemeConfig(),
                               ) -> float:
    """
    Compute epsilon for the RDP-DCO allocation scheme.
    
    Args:
        params: Privacy parameters
        config: Scheme configuration parameters
    
    Returns:
        Computed epsilon value
    """
    params.validate()
    if params.delta is None:
        raise ValueError("Delta must be provided to compute epsilon")
    
    # Use alpha_orders directly from config or generate if not provided
    alpha_orders = config.allocation_RDP_DCO_alpha_orders
    if alpha_orders is None:
        small_alpha_orders = np.linspace(1.001, 2, 20)
        alpha_orders = np.arange(2, 202)
        large_alpha_orders = np.exp(np.linspace(np.log(202), np.log(10_000), 50)).astype(int)
        alpha_orders = np.concatenate((small_alpha_orders, alpha_orders, large_alpha_orders))

    # Compute RDP and epsilon values
    if config.direction != 'add':
        alpha_RDP = params.num_epochs*np.array([allocation_RDP_DCO_remove(params.sigma, params.num_steps, params.num_selected, alpha)
                                         for alpha in alpha_orders])
        alpha_epsilons = alpha_RDP + np.log1p(-1/alpha_orders) - np.log(params.delta * alpha_orders)/(alpha_orders-1)
        epsilon_remove = np.min(alpha_epsilons)
        used_alpha_remove = alpha_orders[np.argmin(alpha_epsilons)]
    if config.direction != 'remove':
        alpha_RDP = params.num_epochs*np.array([allocation_RDP_DCO_add(params.sigma, params.num_steps, params.num_selected, alpha)
                                         for alpha in alpha_orders])
        alpha_epsilons = alpha_RDP + np.log1p(-1/alpha_orders) - np.log(params.delta * alpha_orders)/(alpha_orders-1)
        epsilon_add = np.min(alpha_epsilons)
        used_alpha_add = alpha_orders[np.argmin(alpha_epsilons)]

    # Determine the epsilon and used alpha based on the direction
    if config.direction == 'add':
        epsilon = epsilon_add
        used_alpha = used_alpha_add
    elif config.direction == 'remove':
        epsilon = epsilon_remove
        used_alpha = used_alpha_remove
    else:
        epsilon = max(epsilon_add, epsilon_remove)
        used_alpha = used_alpha_add if epsilon_add > epsilon_remove else used_alpha_remove

    # Check for potential alpha overflow or underflow
    if used_alpha == alpha_orders[-1]:
        print(f'Potential alpha overflow! used alpha: {used_alpha} which is the maximal alpha')
    if used_alpha == alpha_orders[0]:
        print(f'Potential alpha underflow! used alpha: {used_alpha} which is the minimal alpha')
    if config.print_alpha:
        print(f'sigma: {params.sigma}, delta: {params.delta}, num_steps: {params.num_steps}, num_selected: {params.num_selected}, num_epochs: {params.num_epochs}, used_alpha: {used_alpha}')
    return epsilon

def allocation_delta_RDP_DCO(params: PrivacyParams,
                             config: SchemeConfig = SchemeConfig(),
                             ) -> float:
    """
    Compute delta for the RDP-DCO allocation scheme.
    
    Args:
        params: Privacy parameters
        config: Scheme configuration parameters
        
    Returns:
        Computed delta value
    """
    params.validate()
    if params.epsilon is None:
        raise ValueError("Epsilon must be provided to compute delta")
    
    # Use alpha_orders directly from config or generate if not provided
    alpha_orders = config.allocation_RDP_DCO_alpha_orders
    if alpha_orders is None:
        small_alpha_orders = np.linspace(1.001, 2, 20)
        alpha_orders = np.arange(2, 202)
        large_alpha_orders = np.exp(np.linspace(np.log(202), np.log(10_000), 50)).astype(int)
        alpha_orders = np.concatenate((small_alpha_orders, alpha_orders, large_alpha_orders))

    # Compute RDP and epsilon values
    if config.direction != 'add':
        alpha_RDP = params.num_epochs*np.array([allocation_RDP_DCO_remove(params.sigma, params.num_steps, params.num_selected, alpha)
                                         for alpha in alpha_orders])
        alpha_deltas = np.exp((alpha_orders-1) * (alpha_RDP - params.epsilon))*(1-1/alpha_orders)**alpha_orders / (alpha_orders-1)
        delta_remove = np.min(alpha_deltas)
        used_alpha_remove = alpha_orders[np.argmin(alpha_deltas)]
    if config.direction != 'remove':
        alpha_RDP = params.num_epochs*np.array([allocation_RDP_DCO_add(params.sigma, params.num_steps, params.num_selected, alpha)
                                         for alpha in alpha_orders])
        alpha_deltas = np.exp((alpha_orders-1) * (alpha_RDP - params.epsilon))*(1-1/alpha_orders)**alpha_orders / (alpha_orders-1)
        delta_add = np.min(alpha_deltas)
        used_alpha_add = alpha_orders[np.argmin(alpha_deltas)]

    # Determine the epsilon and used alpha based on the direction
    if config.direction == 'add':
        delta = delta_add
        used_alpha = used_alpha_add
    elif config.direction == 'remove':
        delta = delta_remove
        used_alpha = used_alpha_remove
    else:
        delta = max(delta_add, delta_remove)
        used_alpha = used_alpha_add if delta_add > delta_remove else used_alpha_remove

    # Check for potential alpha overflow or underflow
    if used_alpha == alpha_orders[-1]:
        print(f'Potential alpha overflow! used alpha: {used_alpha} which is the maximal alpha')
    if used_alpha == alpha_orders[0]:
        print(f'Potential alpha underflow! used alpha: {used_alpha} which is the minimal alpha')
    if config.print_alpha:
        print(f'sigma: {params.sigma}, epsilon: {params.epsilon}, num_steps: {params.num_steps}, num_selected: {params.num_selected}, num_epochs: {params.num_epochs}, used_alpha: {used_alpha}')
    return delta