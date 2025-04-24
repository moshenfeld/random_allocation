from functools import cache
from numba import jit
from typing import List, Tuple
import numpy as np
import math

from random_allocation.other_schemes.local import bin_search


# ==================== Add ====================
def allocation_epsilon_rdp_add(sigma: float, 
                               delta: float, 
                               num_steps: int, 
                               num_epochs: int, 
                               print_alpha: bool) -> float:
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
    # Define alpha orders for RDP computation
    small_alpha_orders = np.linspace(1.001, 2, 20)
    alpha_orders = np.arange(2, 202)
    large_alpha_orders = np.exp(np.linspace(np.log(202), np.log(10_000), 50)).astype(int)
    alpha_orders = np.concatenate((small_alpha_orders, alpha_orders, large_alpha_orders))

    # Compute RDP and epsilon values
    alpha_rdp = num_epochs * (alpha_orders + num_steps - 1) / (2 * num_steps * sigma**2)
    alpha_epsilons = alpha_rdp + np.log1p(-1 / alpha_orders) - np.log(delta * alpha_orders) / (alpha_orders - 1)
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
def allocation_rdp_remove(alpha: int, sigma: float, num_steps: int) -> float:
    """
    Compute the RDP of the allocation scheme in the emove direction.
    This function is based on the first part of Corollary 6.2,
    """
    partitions = generate_partitions(n=alpha, max_size=num_steps)
    exp_terms = [compute_exp_term(partition=partition, alpha=alpha, num_steps=num_steps, sigma=sigma) for partition in partitions]

    max_val = max(exp_terms)
    log_sum = np.log(sum(np.exp(term - max_val) for term in exp_terms))

    return (log_sum - alpha*(1/(2*sigma**2) + np.log(num_steps)) + max_val) / (alpha-1)

def allocation_epsilon_rdp_remove(sigma: float,
                                  delta: float,
                                  num_steps:int,
                                  num_epochs:int,
                                  alpha_orders: List[int],
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
        alpha_orders (List[int]): List of alpha orders for RDP computation.
        print_alpha (bool): Whether to print the alpha value used.
    """
    alpha = alpha_orders[0]
    alpha_rdp = allocation_rdp_remove(alpha, sigma, num_steps)*num_epochs
    epsilon = alpha_rdp + math.log1p(-1/alpha) - math.log(delta * alpha)/(alpha-1)
    used_alpha = alpha
    for alpha in alpha_orders:
        alpha_rdp = allocation_rdp_remove(alpha, sigma, num_steps)*num_epochs
        if alpha_rdp > epsilon:
            break
        else:
            new_eps = alpha_rdp + math.log1p(-1/alpha) - math.log(delta * alpha)/(alpha-1)
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
def allocation_epsilon_rdp(sigma: float,
                           delta: float,
                           num_steps: int,
                           num_selected: int,
                           num_epochs: int,
                           direction: str = 'both',
                           min_alpha: int = 2,
                           max_alpha: int = 50,
                           print_alpha: bool = False,
                           ) -> float:
    """
    Compute the epsilon value of the allocation scheme using Rényi Differential Privacy (RDP).
    This function can compute epsilon for both the add and remove directions, or maximum of both.
    Args:
        sigma (float): Gaussian noise scale.
        delta (float): Target delta value for differential privacy.
        num_steps (int): Number of steps in the allocation scheme.  
        num_selected (int): Number of selected elements in the allocation scheme.
        num_epochs (int): Number of epochs.
        direction (str): Direction of the allocation scheme ('add', 'remove', or 'both').
        min_alpha (int): Minimum alpha value for RDP computation.
        max_alpha (int): Maximum alpha value for RDP computation.
        print_alpha (bool): Whether to print the alpha value used.
    """
    num_steps_per_round = int(np.ceil(num_steps/num_selected))
    num_rounds = int(np.ceil(num_steps/num_steps_per_round))
    if direction != 'add':
        alpha_orders = np.arange(min_alpha, max_alpha+1)
        epsilon_remove = allocation_epsilon_rdp_remove(sigma=sigma, delta=delta, num_steps=num_steps_per_round,
                                                       num_epochs=num_rounds*num_epochs, alpha_orders=alpha_orders, print_alpha=print_alpha)
    if direction != 'remove':
        epsilon_add = allocation_epsilon_rdp_add(sigma=sigma, delta=delta, num_steps=num_steps_per_round,
                                                 num_epochs=num_rounds*num_epochs, print_alpha=print_alpha)
    if direction == 'add':
        return epsilon_add
    if direction == 'remove':
        return epsilon_remove
    return max(epsilon_remove, epsilon_add)

def allocation_delta_rdp(sigma: float,
                         epsilon: float,
                         num_steps: int,
                         num_selected: int,
                         num_epochs: int,
                         direction: str = 'both',
                         min_alpha: int = 2,
                         max_alpha: int = 50,
                         delta_tolerance: float = 1e-15,
                         ) -> float:
    """
    Compute the delta value of the allocation scheme using Rényi Differential Privacy (RDP).
    This function can compute delta for both the add and remove directions, or maximum of both.
    Args:
        sigma (float): Gaussian noise scale.
        epsilon (float): Target epsilon value for differential privacy.
        num_steps (int): Number of steps in the allocation scheme.
        num_selected (int): Number of selected elements in the allocation scheme.
        num_epochs (int): Number of epochs.
        direction (str): Direction of the allocation scheme ('add', 'remove', or 'both').
        min_alpha (int): Minimum alpha value for RDP computation.
        max_alpha (int): Maximum alpha value for RDP computation.
        delta_tolerance (float): Tolerance for delta computation.
    """
    if direction != 'add':
        delta_remove =  bin_search(lambda delta: allocation_epsilon_rdp(sigma=sigma, delta=delta, num_steps=num_steps,
                                                                        num_selected=num_selected, num_epochs=num_epochs, direction='remove', min_alpha=min_alpha, max_alpha=max_alpha, print_alpha=False),
                      lower=0, upper=1, target=epsilon, tolerance=delta_tolerance, increasing=False)
    if direction != 'remove':
        delta_add =  bin_search(lambda delta: allocation_epsilon_rdp(sigma=sigma, delta=delta, num_steps=num_steps,
                                                                     num_selected=num_selected, num_epochs=num_epochs, direction='add', min_alpha=min_alpha, max_alpha=max_alpha, print_alpha=False),
                      lower=0, upper=1, target=epsilon, tolerance=delta_tolerance, increasing=False)
    if direction == 'add':
        return delta_add
    if direction == 'remove':
        return delta_remove
    return max(delta_add, delta_remove)