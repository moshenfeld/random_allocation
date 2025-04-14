from functools import cache, lru_cache
from numba import jit
from typing import List, Tuple
import math
import numpy as np
from random_allocation.other_schemes.local import bin_search

@cache
def generate_partitions(n: int, max_size: int) -> List[List[Tuple[int, ...]]]:
    ''' Generate all integer partitions of [1, ..., n] with a maximum number of elements in the partition '''
    partitions = [[] for _ in range(n + 1)]
    partitions[0].append(())

    for i in range(1, n):
        partitions[i] = generate_partitions(n=i, max_size=max_size)
    for j in range(n, 0, -1):
        for p in partitions[n - j]:
            if (not p or j <= p[0]) and len(p) < max_size:  # Ensure descending order
                partitions[n].append((j,) + p)
    return partitions[n]

@jit(nopython=True)
def log_factorial(n: int) -> float:
    ''' Compute log(n!) '''
    if n <= 1:
        return 0.0
    return np.sum(np.log(np.arange(1, n + 1)))

@jit(nopython=True)
def log_factorial_range(n: int, m:int) -> float:
    ''' Compute log(n! / (n-m)!) '''
    if n <= 1:
        return 0.0
    return np.sum(np.log(np.arange(n - m + 1, n + 1)))

@jit(nopython=True)
def calc_partition_sum_square(arr: Tuple[int, ...]) -> float:
    ''' Compute the sum of squares of an array. e.g. (1, 2, 3) -> 1^2 + 2^2 + 3^2 '''
    result = 0.0
    for x in arr:
        result += x * x
    return result

@lru_cache(maxsize=None)
def calc_partition_sum_square_cached(arr: Tuple[int, ...]) -> float:
    ''' Compute the sum of squares of an array. e.g. (1, 2, 3) -> 1^2 + 2^2 + 3^2 '''
    return calc_partition_sum_square(arr=arr)

@jit(nopython=True)
def calc_log_multinomial(partition: Tuple[int, ...], n: int) -> float:
    ''' Compute the log of the multinomial of n over its partition'''
    log_prod_factorial = 0.0
    for p in partition:
        log_prod_factorial += log_factorial(n=p)

    return log_factorial(n=n) - log_prod_factorial

@lru_cache(maxsize=None)
def calc_log_multinomial_cached(partition: Tuple[int, ...], n: int) -> float:
    ''' Compute the log of the multinomial of n over its partition'''
    return calc_log_multinomial(partition=partition, n=n)

@jit(nopython=True)
def calc_counts_log_multinomial(partition: Tuple[int, ...], n: int) -> float:
    ''' Compute the counts of each unique integer in an array and calculate multinomial '''
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

@lru_cache(maxsize=None)
def calc_counts_log_multinomial_cached(partition: Tuple[int, ...], n: int) -> float:
    ''' Compute the counts of each unique integer in an array and calculate multinomial '''
    return calc_counts_log_multinomial(partition=partition, n=n)

@cache
def compute_exp_term(partition: Tuple[int, ...], alpha: int, num_steps: int, sigma: float) -> float:
    ''' Compute the exponent term of the sum '''
    counts_log_multinomial = calc_counts_log_multinomial_cached(partition=partition, n=num_steps)
    partition_log_multinomial = calc_log_multinomial_cached(partition=partition, n=alpha)
    partition_sum_square = calc_partition_sum_square_cached(arr=partition) / (2 * sigma**2)
    return counts_log_multinomial + partition_log_multinomial + partition_sum_square

@cache
def allocation_rdp(alpha: int, sigma: float, num_steps: int) -> float:
    ''' Compute the RDP of the allocation mechanism '''
    partitions = generate_partitions(n=alpha, max_size=num_steps)
    exp_terms = [compute_exp_term(partition=partition, alpha=alpha, num_steps=num_steps, sigma=sigma) for partition in partitions]

    max_val = max(exp_terms)
    log_sum = np.log(sum(np.exp(term - max_val) for term in exp_terms))

    return (log_sum - alpha*(1/(2*sigma**2) + np.log(num_steps)) + max_val) / (alpha-1)

@cache
def allocation_epsilon_rdp(sigma: float,
                           delta: float,
                           num_steps: int,
                           num_selected: int,
                           num_epochs: int,
                           min_alpha: int = 2,
                           max_alpha: int = 50,
                           print_alpha: bool = False,
                           ) -> float:
    num_steps_per_round = np.ceil(num_steps/num_selected).astype(int)
    num_rounds = np.ceil(num_steps/num_steps_per_round).astype(int)
    alpha_rdp = allocation_rdp(min_alpha, sigma, num_steps_per_round)*num_rounds*num_epochs
    epsilon = alpha_rdp + math.log1p(-1/min_alpha) - math.log(delta * min_alpha)/(min_alpha-1)
    used_alpha = min_alpha
    alpha = int(min_alpha+1)
    surpased_rdp = False
    while alpha <= max_alpha and not surpased_rdp:
        alpha_rdp = allocation_rdp(alpha, sigma, num_steps_per_round)*num_rounds*num_epochs
        if alpha_rdp > epsilon:
            surpased_rdp = True
            break
        else:
            new_eps = alpha_rdp + math.log1p(-1/alpha) - math.log(delta * alpha)/(alpha-1)
            if new_eps < epsilon:
                epsilon = new_eps
                used_alpha = alpha
            alpha += 1
    if used_alpha == max_alpha:
        print(f'Potential alpha overflow! used alpha: {used_alpha} which is the maximal alpha')
        # return np.inf
    if used_alpha == min_alpha:
        print(f'Potential alpha underflow! used alpha: {used_alpha} which is the minimal alpha')
    if print_alpha:
        print(f'Used alpha: {used_alpha}')
    return epsilon

@cache
def allocation_delta_rdp(sigma: float,
                         epsilon: float,
                         num_steps: int,
                         num_selected: int,
                         num_epochs: int,
                         min_alpha: int = 2,
                         max_alpha: int = 50,
                         ) -> float:
    return bin_search(lambda delta: allocation_epsilon_rdp(sigma=sigma, delta=delta, num_steps=num_steps, num_selected=num_selected,
                                                           num_epochs=num_epochs, min_alpha=min_alpha, max_alpha=max_alpha),
                      0, 1, epsilon, increasing=False)