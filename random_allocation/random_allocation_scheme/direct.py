from functools import cache
from numba import jit
from typing import List, Tuple
import numpy as np
import math

from random_allocation.comparisons.utils import search_function_with_bounds, FunctionType
from random_allocation.other_schemes.local import Gaussian_epsilon, Gaussian_delta
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig

# ==================== Add ====================
def allocation_epsilon_direct_add(sigma: float,
                                  delta: float,
                                  num_steps: int,
                                  num_epochs: int,
                                  ) -> float:
    return Gaussian_epsilon(sigma=sigma*math.sqrt(num_steps/(num_epochs)), delta=delta) + (1-1.0/num_steps)/(2*sigma**2)

def allocation_delta_direct_add(sigma: float,
                                epsilon: float,
                                num_steps: int,
                                num_epochs: int,
                                ) -> float:
    return Gaussian_delta(sigma=sigma*math.sqrt(num_steps/(num_epochs)), epsilon=epsilon - (1-1.0/num_steps)/(2*sigma**2))

# ==================== Remove ====================
@cache
def generate_partitions(n: int, max_size: int) -> List[List[Tuple[int, ...]]]:
    """
    Generate all integer partitions of [1, ..., n] with a maximum number of elements in the partition.
    """
    partitions = [[] for _ in range(n + 1)]
    partitions[0].append(())

    for i in range(1, n):
        partitions[i] = generate_partitions(n=i, max_size=max_size)
    for j in range(n, 0, -1):
        for p in partitions[n - j]:
            if (not p or j <= p[0]) and len(p) < max_size:  # Ensure descending order
                partitions[n].append((j,) + p)
    return partitions[n]

@jit(nopython=True, cache=True)
def log_factorial(n: int) -> float:
    """
    Compute the natural logarithm of n!.
    """
    if n <= 1:
        return 0.0
    return np.sum(np.log(np.arange(1, n + 1)))

@jit(nopython=True, cache=True)
def log_factorial_range(n: int, m: int) -> float:
    """
    Compute the natural logarithm of (n! / (n-m)!).
    """
    if n <= 1:
        return 0.0
    return np.sum(np.log(np.arange(n - m + 1, n + 1)))

@jit(nopython=True, cache=True)
def calc_partition_sum_square(arr: Tuple[int, ...]) -> float:
    """
    Compute the sum of squares of an array.
    """
    result = 0.0
    for x in arr:
        result += x * x
    return result

@jit(nopython=True, cache=True)
def calc_log_multinomial(partition: Tuple[int, ...], n: int) -> float:
    """
    Compute the log of the multinomial coefficient for a given partition.

    """
    log_prod_factorial = 0.0
    for p in partition:
        log_prod_factorial += log_factorial(n=p)
    return log_factorial(n=n) - log_prod_factorial

@jit(nopython=True, cache=True)
def calc_counts_log_multinomial(partition: Tuple[int, ...], n: int) -> float:
    """
    Compute the counts of each unique integer in a partition and calculate the multinomial coefficient.
    """
    sum_partition = sum(partition)

    # Count frequencies
    counts = np.zeros(sum_partition + 1, dtype=np.int64)
    for x in partition:
        counts[x] += 1
    sum_counts = sum(counts)

    # Compute multinomial
    log_counts_factorial = 0.0
    for i in range(1, sum_partition + 1):
        if counts[i] > 0:
            log_counts_factorial += log_factorial(n=counts[i])

    return log_factorial_range(n=n, m=sum_counts) - log_counts_factorial

@jit(nopython=True, cache=True)
def compute_exp_term(partition: Tuple[int, ...], alpha: int, num_steps: int, sigma: float) -> float:
    """
    Compute the exponent term that is summed up inside the log term in the first of Corollary 6.2.
    """
    counts_log_multinomial = calc_counts_log_multinomial(partition=partition, n=num_steps)
    partition_log_multinomial = calc_log_multinomial(partition=partition, n=alpha)
    partition_sum_square = calc_partition_sum_square(arr=partition) / (2 * sigma**2)
    return counts_log_multinomial + partition_log_multinomial + partition_sum_square

@cache
def allocation_RDP_remove(alpha: int, sigma: float, num_steps: int) -> float:
    """
    Compute the RDP of the allocation scheme in the emove direction.
    This function is based on the first part of Corollary 6.2,
    """
    partitions = generate_partitions(n=alpha, max_size=num_steps)
    exp_terms = [compute_exp_term(partition=partition, alpha=alpha, num_steps=num_steps, sigma=sigma) for partition in partitions]

    max_val = max(exp_terms)
    log_sum = np.log(sum(np.exp(term - max_val) for term in exp_terms))

    return (log_sum - alpha*(1/(2*sigma**2) + np.log(num_steps)) + max_val) / (alpha-1)

def allocation_epsilon_direct_remove(sigma: float,
                                  delta: float,
                                  num_steps:int,
                                  num_epochs:int,
                                  alpha_orders,
                                  print_alpha: bool,
                                  ) -> float:
    """
    Compute the epsilon value of the allocation scheme in the remove direction using Rényi Differential Privacy (RDP).
    This function is based on Lemma 2.4, and utilizes the improvement stated in Claim 6.4.
    Args:       
        sigma (float): Gaussian noise scale.
        delta (float): Target delta value for differential privacy.
        num_steps (int): Number of steps in the allocation scheme.
        num_epochs (int): Number of epochs.
        alpha_orders: Array of alpha orders for RDP computation.
        print_alpha (bool): Whether to print the alpha value used.
    """
    alpha = alpha_orders[0]
    alpha_RDP = allocation_RDP_remove(alpha, sigma, num_steps)*num_epochs
    epsilon = alpha_RDP + math.log1p(-1/alpha) - math.log(delta * alpha)/(alpha-1)
    used_alpha = alpha
    for alpha in alpha_orders:
        alpha_RDP = allocation_RDP_remove(alpha, sigma, num_steps)*num_epochs
        if alpha_RDP > epsilon:
            break
        else:
            new_eps = alpha_RDP + math.log1p(-1/alpha) - math.log(delta * alpha)/(alpha-1)
            if new_eps < epsilon:
                epsilon = new_eps
                used_alpha = alpha
    
    if used_alpha == alpha_orders[-1]:
        print(f'Potential alpha overflow! used alpha: {used_alpha} which is the maximal alpha')
    if used_alpha == alpha_orders[0]:
        print(f'Potential alpha underflow! used alpha: {used_alpha} which is the minimal alpha')
    if print_alpha:
        print(f'sigma: {sigma}, delta: {delta}, num_steps: {num_steps}, num_epochs: {num_epochs}, used_alpha: {used_alpha}')
    return epsilon

# ==================== Both ====================
def allocation_epsilon_direct(params: PrivacyParams,
                           config: SchemeConfig = SchemeConfig(),
                           ) -> float:
    """
    Compute the epsilon value of the allocation scheme using Rényi Differential Privacy (RDP).
    This function can compute epsilon for both the add and remove directions, or maximum of both.
    
    Args:
        params: Privacy parameters (must include delta)
        config: Scheme configuration parameters
    
    Returns:
        Computed epsilon value
    """
    params.validate()
    if params.delta is None:
        raise ValueError("Delta must be provided to compute epsilon")
    
    num_steps_per_round = int(np.ceil(params.num_steps/params.num_selected))
    num_rounds = int(np.ceil(params.num_steps/num_steps_per_round))
    
    if config.direction != 'add':
        epsilon_remove = allocation_epsilon_direct_remove(
            sigma=params.sigma, 
            delta=params.delta, 
            num_steps=num_steps_per_round,
            num_epochs=num_rounds*params.num_epochs, 
            alpha_orders=config.allocation_direct_alpha_orders, 
            print_alpha=config.print_alpha
        )
    
    if config.direction != 'remove':
        epsilon_add = allocation_epsilon_direct_add(
            sigma=params.sigma, 
            delta=params.delta, 
            num_steps=num_steps_per_round,
            num_epochs=num_rounds*params.num_epochs
        )
    
    if config.direction == 'add':
        return epsilon_add
    if config.direction == 'remove':
        return epsilon_remove
    return max(epsilon_remove, epsilon_add)

def allocation_delta_direct(params: PrivacyParams,
                         config: SchemeConfig = SchemeConfig(),
                         ) -> float:
    """
    Compute the delta value of the allocation scheme using Rényi Differential Privacy (RDP).
    This function can compute delta for both the add and remove directions, or maximum of both.
    
    Args:
        params: Privacy parameters (must include epsilon)
        config: Scheme configuration parameters
    
    Returns:
        Computed delta value
    """
    params.validate()
    if params.epsilon is None:
        raise ValueError("Epsilon must be provided to compute delta")
    
    if config.direction != 'add':
        # Create a copy of params with epsilon=None to use in optimization function
        params_copy = PrivacyParams(
            sigma=params.sigma,
            num_steps=params.num_steps,
            num_selected=params.num_selected,
            num_epochs=params.num_epochs,
            epsilon=None,
            delta=None  # This will be set by the optimization function
        )
        
        def optimization_func(delta):
            params_copy.delta = delta
            # Use 'remove' specifically for this optimization
            remove_config = SchemeConfig(
                direction='remove',
                allocation_direct_alpha_orders=config.allocation_direct_alpha_orders,
                print_alpha=False,
                delta_tolerance=config.delta_tolerance
            )
            return allocation_epsilon_direct(params=params_copy, config=remove_config)
            
        delta_remove = search_function_with_bounds(
            func=optimization_func, 
            y_target=params.epsilon, 
            bounds=(config.delta_tolerance, 1-config.delta_tolerance), 
            tolerance=config.delta_tolerance,
            function_type=FunctionType.DECREASING
        )
    
    if config.direction != 'remove':
        num_steps_per_round = int(np.ceil(params.num_steps/params.num_selected))
        num_rounds = int(np.ceil(params.num_steps/num_steps_per_round))
        delta_add = allocation_delta_direct_add(
            sigma=params.sigma, 
            epsilon=params.epsilon, 
            num_steps=num_steps_per_round,
            num_epochs=params.num_epochs*num_rounds
        )
    
    if config.direction == 'add':
        return delta_add
    if config.direction == 'remove':
        return delta_remove
    return max(delta_add, delta_remove)

# ==================== RDP based Add ====================
def allocation_RDP_add(sigma: float, 
                       num_steps: int, 
                       num_epochs: int,
                       ) -> Tuple[np.ndarray, np.ndarray]:
    small_alpha_orders = np.linspace(1.001, 2, 20)
    alpha_orders = np.arange(2, 202)
    large_alpha_orders = np.exp(np.linspace(np.log(202), np.log(10_000), 50)).astype(int)
    alpha_orders = np.concatenate((small_alpha_orders, alpha_orders, large_alpha_orders))

    # Compute RDP values
    alpha_RDP = num_epochs * (alpha_orders + num_steps - 1) / (2 * num_steps * sigma**2)
    return alpha_orders, alpha_RDP

def allocation_epsilon_RDP_add(sigma: float,
                               delta: float,
                               num_steps: int,
                               num_epochs: int,
                               print_alpha: bool = False,
                               ) -> float:
    """
    Compute the epsilon value of the allocation scheme in the add direction using Rényi Differential Privacy (RDP).
    This function is based on the second part of Corollary 6.2, combined with Lemma 2.4.

    Args:
        sigma (float): Gaussian noise scale.
        delta (float): Target delta value for differential privacy.
        num_steps (int): Number of steps in the allocation scheme.
        num_epochs (int): Number of epochs.
        print_alpha (bool): Whether to print the alpha value used.
    """
    # Compute RDP and epsilon values
    alpha_orders, alpha_RDP = allocation_RDP_add(sigma, num_steps, num_epochs)
    alpha_epsilons = alpha_RDP + np.log1p(-1 / alpha_orders) - np.log(delta * alpha_orders) / (alpha_orders - 1)
    epsilon = np.min(alpha_epsilons)
    used_alpha = alpha_orders[np.argmin(alpha_epsilons)]

    # Check for potential alpha overflow or underflow
    if used_alpha == alpha_orders[-1]:
        print(f'Potential alpha overflow! used alpha: {used_alpha} which is the maximal alpha')
    if used_alpha == alpha_orders[0]:
        print(f'Potential alpha underflow! used alpha: {used_alpha} which is the minimal alpha')

    # Optionally print the alpha value used
    if print_alpha:
        print(f'sigma: {sigma}, delta: {delta}, num_steps: {num_steps}, num_epochs: {num_epochs}, used_alpha: {used_alpha}')
    
    return epsilon

def allocation_delta_RDP_add(sigma: float,
                             epsilon: float,
                             num_steps: int,
                             num_selected: int,
                             num_epochs: int,
                             print_alpha: bool,
                             ) -> float:
    """
    Compute the privacy profile of the allocation scheme in the add direction using Rényi Differential Privacy (RDP).
    This function is based on the second part of Corollary 6.2, combined with Lemma 2.4.

    Args:
        sigma (float): Gaussian noise scale.
        epsilon (float): Target epsilon value for differential privacy.
        num_steps (int): Number of steps in the allocation scheme.
        num_epochs (int): Number of epochs.
        print_alpha (bool): Whether to print the alpha value used.
    """
    num_steps_per_round = int(np.ceil(num_steps/num_selected))
    num_rounds = int(np.ceil(num_steps/num_steps_per_round))

    # Compute RDP and epsilon values
    alpha_orders, alpha_RDP = allocation_RDP_add(sigma, num_steps_per_round, num_epochs*num_rounds)
    alpha_deltas = np.exp((alpha_orders-1) * (alpha_RDP - epsilon))*(1-1/alpha_orders)**alpha_orders / (alpha_orders-1)
    delta = np.min(alpha_deltas)
    used_alpha = alpha_orders[np.argmin(alpha_deltas)]

    # Check for potential alpha overflow or underflow
    if used_alpha == alpha_orders[-1]:
        print(f'Potential alpha overflow! used alpha: {used_alpha} which is the maximal alpha')
    if used_alpha == alpha_orders[0]:
        print(f'Potential alpha underflow! used alpha: {used_alpha} which is the minimal alpha')

    # Optionally print the alpha value used
    if print_alpha:
        print(f'sigma: {sigma}, epsilon: {epsilon}, num_steps: {num_steps}, num_selected: {num_selected}, num_epochs: {num_epochs}, used_alpha: {used_alpha}')
    
    return delta