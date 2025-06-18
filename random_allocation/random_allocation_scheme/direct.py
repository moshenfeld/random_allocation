# Standard library imports
from functools import lru_cache
import math
from typing import List, Tuple, Callable

# Third-party imports
from numba import jit
import numpy as np
from numpy.typing import NDArray

# Local application imports
from random_allocation.comparisons.utils import search_function_with_bounds, FunctionType, BoundType
from random_allocation.other_schemes.local import Gaussian_epsilon, Gaussian_delta
from random_allocation.comparisons.definitions import PrivacyParams, SchemeConfig, Direction, Verbosity
from random_allocation.random_allocation_scheme.random_allocation_utils import print_alpha

# Type aliases
Partition = Tuple[int, ...]
NumericFunction = Callable[[float], float]

# ==================== Add ====================
def allocation_epsilon_direct_add(sigma: float, delta: float, num_steps: int, num_epochs: int) -> float:
    """
    Compute the epsilon value of the allocation scheme in the add direction.
    
    Args:
        sigma: Gaussian noise scale
        delta: Target delta value for differential privacy
        num_steps: Number of steps in the allocation scheme
        num_epochs: Number of epochs
        
    Returns:
        Computed epsilon value
    """
    epsilon = Gaussian_epsilon(sigma=sigma*math.sqrt(num_steps/num_epochs), delta=delta)
    epsilon += (1-1.0/num_steps)/(2*sigma**2)
    return float(epsilon)

def allocation_delta_direct_add(sigma: float, epsilon: float, num_steps: int, num_epochs: int) -> float:
    """
    Compute the delta value of the allocation scheme in the add direction.
    
    Args:
        sigma: Gaussian noise scale
        epsilon: Target epsilon value for differential privacy
        num_steps: Number of steps in the allocation scheme
        num_epochs: Number of epochs
        
    Returns:
        Computed delta value
    """
    result = Gaussian_delta(sigma=sigma*math.sqrt(num_steps/(num_epochs)), epsilon=epsilon - (1-1.0/num_steps)/(2*sigma**2))
    return float(result)

# ==================== Remove ====================
@lru_cache(maxsize=None)
def generate_partitions(n: int, max_size: int) -> List[Tuple[int, ...]]:
    """
    Generate all integer partitions of [1, ..., n] with a maximum number of elements in the partition.
    
    Args:
        n: The number to partition
        max_size: The maximum number of elements in each partition
        
    Returns:
        List of partitions, where each partition is a tuple of integers
    """
    # Convert n to int to handle cases where it's passed as numpy.float64
    n = int(n)
    max_size = int(max_size)
    
    partitions: List[List[Tuple[int, ...]]] = [[] for _ in range(n + 1)]
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
    return float(np.sum(np.log(np.arange(1, n + 1))))

@jit(nopython=True, cache=True)
def log_factorial_range(n: int, m: int) -> float:
    """
    Compute the natural logarithm of (n! / (n-m)!).
    """
    if n <= 1:
        return 0.0
    return float(np.sum(np.log(np.arange(n - m + 1, n + 1))))

@jit(nopython=True, cache=True)
def calc_partition_sum_square(arr: Tuple[int, ...]) -> float:
    """
    Compute the sum of squares of an array.
    """
    result = 0.0
    for x in arr:
        result += x * x
    return float(result)

@jit(nopython=True, cache=True)
def calc_log_multinomial(partition: Tuple[int, ...], n: int) -> float:
    """
    Compute the log of the multinomial coefficient for a given partition.

    """
    log_prod_factorial = 0.0
    for p in partition:
        log_prod_factorial += log_factorial(n=p)
    return float(log_factorial(n=n) - log_prod_factorial)

@jit(nopython=True, cache=True)
def calc_counts_log_multinomial(partition: Tuple[int, ...], n: int) -> float:
    """
    Compute the counts of each unique integer in a partition and calculate the multinomial coefficient.
    """
    sum_partition = sum(partition)

    # Count frequencies
    counts = np.zeros(sum_partition + 1, dtype=np.int64)  # type: np.ndarray
    for x in partition:
        counts[x] += 1
    sum_counts = sum(counts)

    # Compute multinomial
    log_counts_factorial = 0.0
    for i in range(1, sum_partition + 1):
        if counts[i] > 0:
            log_counts_factorial += log_factorial(n=counts[i])

    return float(log_factorial_range(n=n, m=sum_counts) - log_counts_factorial)

@jit(nopython=True, cache=True)
def compute_exp_term(partition: Tuple[int, ...], alpha: int, num_steps: int, sigma: float) -> float:
    """
    Compute the exponent term that is summed up inside the log term in the first of Corollary 6.2.
    """
    counts_log_multinomial = calc_counts_log_multinomial(partition=partition, n=num_steps)
    partition_log_multinomial = calc_log_multinomial(partition=partition, n=alpha)
    partition_sum_square = calc_partition_sum_square(arr=partition) / (2 * sigma**2)
    return float(counts_log_multinomial + partition_log_multinomial + partition_sum_square)

@lru_cache(maxsize=None)
def allocation_RDP_remove(alpha: int, sigma: float, num_steps: int) -> float:
    """
    Compute the RDP value of the allocation scheme in the remove direction for a given alpha.
    
    Args:
        alpha: Alpha order for RDP
        sigma: Gaussian noise scale
        num_steps: Number of steps in the allocation scheme
        
    Returns:
        Computed RDP value
    """
    assert isinstance(alpha, int), "alpha must be an integer"
    assert isinstance(num_steps, int), "num_steps must be an integer"
    assert alpha > 1, "alpha must be > 1"
        
    partitions = generate_partitions(n=alpha, max_size=num_steps)
    exp_terms = [compute_exp_term(partition=partition, alpha=alpha, num_steps=num_steps, sigma=sigma) for partition in partitions]

    max_val = max(exp_terms)
    log_sum = np.log(sum(np.exp(term - max_val) for term in exp_terms))

    return float((log_sum - alpha*(1/(2*sigma**2) + np.log(num_steps)) + max_val) / (alpha - 1))

def allocation_epsilon_direct_remove(sigma: float,
                                     delta: float,
                                     num_steps: int,
                                     num_epochs: int,
                                     alpha_orders: List[int],
                                     verbosity: Verbosity = Verbosity.NONE,
                                     ) -> float:
    """
    Compute the epsilon value of the allocation scheme in the remove direction using Rényi Differential Privacy (RDP).
    This function is based on Lemma 2.4, and utilizes the improvement stated in Claim 6.4.
    
    Args:       
        sigma: Gaussian noise scale
        delta: Target delta value for differential privacy
        num_steps: Number of steps in the allocation scheme
        num_epochs: Number of epochs
        alpha_orders: List of alpha orders for RDP computation
        verbosity: Level of verbosity for logging (default: NONE)
        
    Returns:
        Computed epsilon value
    """
    # Ensure alpha_orders has at least one element
    assert len(alpha_orders) > 0, "alpha_orders must have at least one element"
    
    alpha = alpha_orders[0]
    alpha_RDP = allocation_RDP_remove(alpha=alpha, sigma=sigma, num_steps=num_steps) * num_epochs
    epsilon = alpha_RDP + max(math.log1p(-1/alpha) - math.log(delta * alpha)/(alpha-1), 0)
    used_alpha = alpha
    for alpha in alpha_orders:
        alpha_RDP = allocation_RDP_remove(alpha=alpha, sigma=sigma, num_steps=num_steps) * num_epochs
        if alpha_RDP > epsilon:
            break
        else:
            new_eps = alpha_RDP + max(math.log1p(-1/alpha) - math.log(delta * alpha)/(alpha-1), 0)
            if new_eps < epsilon:
                epsilon = new_eps
                used_alpha = alpha
    
    if verbosity != Verbosity.NONE:
        if used_alpha == alpha_orders[-1]:
            print(f'Potential alpha overflow! used alpha: {used_alpha} which is the maximal alpha')
        if used_alpha == alpha_orders[0]:
            print(f'Potential alpha underflow! used alpha: {used_alpha} which is the minimal alpha')
        
    if verbosity == Verbosity.ALL:
        print(f'sigma: {sigma}, num_steps: {num_steps}, num_epochs: {num_epochs}, used_alpha: {used_alpha}')
    
    return float(epsilon)

# ==================== Both ====================
def allocation_epsilon_direct(params: PrivacyParams, config: SchemeConfig, direction: Direction = Direction.BOTH) -> float:
    """
    Compute epsilon for the direct allocation scheme.
    
    Args:
        params: Privacy parameters (must include delta)
        config: Scheme configuration parameters
        direction: The direction of privacy. Can be ADD, REMOVE, or BOTH.
    
    Returns:
        Computed epsilon value
    """
    params.validate()
    if params.delta is None:
        raise ValueError("Delta must be provided to compute epsilon")

    if params.sampling_probability < 1.0:
        raise ValueError('Sampling probability must be 1.0 for allocation direct method')

    num_steps_per_round = int(np.ceil(params.num_steps/params.num_selected))
    num_rounds = int(np.ceil(params.num_steps/num_steps_per_round))

    if direction != Direction.ADD:
        if config.allocation_direct_alpha_orders is None:
            raise ValueError("allocation_direct_alpha_orders must be provided in SchemeConfig for 'remove' direction")
        epsilon_remove = allocation_epsilon_direct_remove(
            sigma=params.sigma, 
            delta=params.delta,
            num_steps=num_steps_per_round, 
            num_epochs=params.num_epochs*num_rounds,
            alpha_orders=config.allocation_direct_alpha_orders,
            verbosity=config.verbosity
        )
    if direction != Direction.REMOVE:
        epsilon_add = allocation_epsilon_direct_add(
            sigma=params.sigma,
            delta=params.delta,
            num_steps=num_steps_per_round,
            num_epochs=params.num_epochs*num_rounds
        )
        
    if direction == Direction.ADD:
        assert epsilon_add is not None, "epsilon_add should be defined"
        return epsilon_add
    if direction == Direction.REMOVE:
        assert epsilon_remove is not None, "epsilon_remove should be defined"
        return epsilon_remove
        
    # Both directions - both should be defined
    assert epsilon_add is not None, "epsilon_add should be defined"
    assert epsilon_remove is not None, "epsilon_remove should be defined"
    return float(max(epsilon_remove, epsilon_add))


def allocation_delta_direct(params: PrivacyParams, config: SchemeConfig, direction: Direction = Direction.BOTH) -> float:
    """
    Compute the delta value of the allocation scheme using Rényi Differential Privacy (RDP).
    This function can compute delta for both the add and remove directions, or maximum of both.
    
    Args:
        params: Privacy parameters (must include epsilon)
        config: Scheme configuration parameters
        direction: The direction of privacy. Can be ADD, REMOVE, or BOTH.
    
    Returns:
        Computed delta value
    """
    params.validate()
    if params.epsilon is None:
        raise ValueError("Epsilon must be provided to compute delta")
    
    if params.sampling_probability < 1.0:
        raise ValueError('Sampling probability must be 1.0 for allocation direct method')

    if direction != Direction.ADD:
        if config.allocation_direct_alpha_orders is None:
            raise ValueError("allocation_direct_alpha_orders must be provided in SchemeConfig for 'remove' or 'both' direction")
        
        result = search_function_with_bounds(
            func=lambda delta: allocation_epsilon_direct(
                params=PrivacyParams(
                    sigma=params.sigma,
                    num_steps=params.num_steps,
                    num_selected=params.num_selected,
                    num_epochs=params.num_epochs,
                    epsilon=None,
                    delta=delta
                ),
                config=config,  # Use the original config
                direction=Direction.REMOVE,
            ),
            y_target=params.epsilon,
            bounds=(2*config.delta_tolerance, 1-2*config.delta_tolerance),
            tolerance=config.delta_tolerance,
            function_type=FunctionType.DECREASING,
            verbosity=config.verbosity,
        )
        
        # Handle case where search function returns None
        if result is None:
            delta_remove = 1.0
        else:
            delta_remove = result
    
    if direction != Direction.REMOVE:
        # Directly calculate delta for the ADD direction
        num_steps_per_round = int(np.ceil(params.num_steps/params.num_selected))
        num_rounds = int(np.ceil(params.num_steps/num_steps_per_round))
        delta_add = allocation_delta_direct_add(
            sigma=params.sigma,
            epsilon=params.epsilon,
            num_steps=num_steps_per_round,
            num_epochs=params.num_epochs*num_rounds
        )
    
    if direction == Direction.ADD:
        return float(delta_add)
    if direction == Direction.REMOVE:
        return float(delta_remove)
    
    # Both directions
    return float(max(delta_remove, delta_add))

# ===================== Legacy RDP add ====================

def allocation_RDP_add(sigma: float, 
                       num_steps: int, 
                       num_epochs: int,
                       alpha_orders: np.ndarray,
                       ) -> np.ndarray:
    # Compute RDP values
    alpha_RDP = num_epochs * (alpha_orders + num_steps - 1) / (2 * num_steps * sigma**2)
    return alpha_RDP

def allocation_delta_RDP_add(params: PrivacyParams, config: SchemeConfig) -> float:
    params.validate()
    if params.epsilon is None:
        raise ValueError("Epsilon must be provided to compute delta")

    if params.sampling_probability < 1.0:
        raise ValueError('Sampling probability must be 1.0 for allocation RDP add method')

    num_steps_per_round = int(np.ceil(params.num_steps/params.num_selected))
    num_rounds = int(np.ceil(params.num_steps/num_steps_per_round))

    alpha_orders = np.array(config.allocation_RDP_DCO_alpha_orders, dtype=np.float64)
    alpha_RDP = allocation_RDP_add(sigma=params.sigma,
                                   num_steps=num_steps_per_round,
                                   num_epochs=num_rounds*params.num_epochs,
                                   alpha_orders=alpha_orders)
    alpha_deltas = np.exp((alpha_orders-1) * (alpha_RDP - params.epsilon))*(1-1/alpha_orders)**alpha_orders / (alpha_orders-1)
    used_alpha = alpha_orders[np.argmin(alpha_deltas)]
    print_alpha(used_alpha, alpha_orders[0], alpha_orders[-1], config.verbosity, "add", params)
    return float(alpha_deltas[used_alpha])

def allocation_epsilon_RDP_add(params: PrivacyParams, config: SchemeConfig) -> float:
    params.validate()
    if params.delta is None:
        raise ValueError("Delta must be provided to compute epsilon")
    
    if params.sampling_probability < 1.0:
        raise ValueError('Sampling probability must be 1.0 for allocation RDP add method')

    num_steps_per_round = int(np.ceil(params.num_steps/params.num_selected))
    num_rounds = int(np.ceil(params.num_steps/num_steps_per_round))

    alpha_orders = np.array(config.allocation_RDP_DCO_alpha_orders, dtype=np.float64)
    alpha_RDP = allocation_RDP_add(sigma=params.sigma,
                                   num_steps=num_steps_per_round,
                                   num_epochs=num_rounds*params.num_epochs,
                                   alpha_orders=alpha_orders)
    alpha_epsilons = alpha_RDP + np.log1p(-1 / alpha_orders) - np.log(params.delta * alpha_orders) / (alpha_orders - 1)
    used_alpha = alpha_orders[np.argmin(alpha_epsilons)]
    print_alpha(used_alpha, alpha_orders[0], alpha_orders[-1], config.verbosity, "add", params)
    return float(alpha_epsilons[used_alpha])