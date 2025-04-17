from functools import cache, lru_cache
from numba import jit
from typing import List, Tuple, Callable
import math
import numpy as np

from random_allocation.other_schemes.local import bin_search
from random_allocation.random_allocation_scheme.inverse import allocation_epsilon_inverse, allocation_delta_inverse

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

def epsilon_from_rdp(sigma: float,
                     delta: float,
                     num_steps:int,
                     num_epochs:int,
                     min_alpha: int,
                     max_alpha: int,
                     rdp_function: Callable,
                     print_alpha: bool,
                     ) -> float:
    alpha_rdp = rdp_function(min_alpha, sigma, num_steps)*num_epochs
    epsilon = alpha_rdp + math.log1p(-1/min_alpha) - math.log(delta * min_alpha)/(min_alpha-1)
    used_alpha = min_alpha
    alpha = int(min_alpha+1)
    surpased_rdp = False
    while alpha <= max_alpha and not surpased_rdp:
        alpha_rdp = rdp_function(alpha, sigma, num_steps)*num_epochs
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
    if used_alpha == min_alpha:
        print(f'Potential alpha underflow! used alpha: {used_alpha} which is the minimal alpha')
    if print_alpha:
        print(f'sigma: {sigma}, delta: {delta}, num_steps: {num_steps}, num_epochs: {num_epochs}, used_alpha: {used_alpha}')
    return epsilon

# @cache
def allocation_rdp_remove(alpha: int, 
                          sigma: float, 
                          num_steps: int
                          ) -> float:
    ''' Compute the RDP of the allocation mechanism '''
    partitions = generate_partitions(n=alpha, max_size=num_steps)
    exp_terms = [compute_exp_term(partition=partition, alpha=alpha, num_steps=num_steps, sigma=sigma) for partition in partitions]

    max_val = max(exp_terms)
    log_sum = np.log(sum(np.exp(term - max_val) for term in exp_terms))

    return (log_sum - alpha*(1/(2*sigma**2) + np.log(num_steps)) + max_val) / (alpha-1)

# @cache
def allocation_epsilon_rdp_remove(sigma: float,
                                  delta: float,
                                  num_steps: int,
                                  num_selected: int,
                                  num_epochs: int,
                                  min_alpha: int,
                                  max_alpha: int,
                                  print_alpha: bool,
                                  ) -> float:
    num_steps_per_round = int(np.ceil(num_steps/num_selected))
    num_rounds = int(np.ceil(num_steps/num_steps_per_round))
    return epsilon_from_rdp(sigma=sigma, delta=delta, num_steps=num_steps_per_round, num_epochs=num_rounds*num_epochs,
                            min_alpha=min_alpha, max_alpha=max_alpha, rdp_function=allocation_rdp_remove, print_alpha=print_alpha)

# @cache
def allocation_delta_rdp_remove(sigma: float,
                                epsilon: float,
                                num_steps: int,
                                num_selected: int,
                                num_epochs: int,
                                min_alpha: int,
                                max_alpha: int,
                                delta_tolerance: float,
                                ) -> float:
    return bin_search(lambda delta: allocation_epsilon_rdp_remove(sigma=sigma, delta=delta, num_steps=num_steps, 
                                                                  num_selected=num_selected, num_epochs=num_epochs, 
                                                                  min_alpha=min_alpha, max_alpha=max_alpha, print_alpha=False),
                      lower=0, upper=1, target=epsilon, tolerance=delta_tolerance, increasing=False)

# @cache
def allocation_rdp_add(alpha: int, 
                       sigma: float, 
                       num_steps: int
                       ) -> float:
    ''' Compute the RDP of the allocation mechanism '''
    partitions = generate_partitions(n=alpha-1, max_size=num_steps)
    exp_terms = [compute_exp_term(partition=partition, alpha=alpha-1, num_steps=num_steps, sigma=sigma)
                 for partition in partitions]

    max_val = max(exp_terms)
    log_sum = np.log(sum(np.exp(term - max_val) for term in exp_terms))

    return (log_sum + max_val) / (alpha-1) - np.log(num_steps) + 1/(2*sigma**2)

# @cache
def allocation_delta_rdp_add(sigma: float,
                             epsilon: float,
                             num_steps: int,
                             num_selected: int,
                             num_epochs: int,
                             min_alpha: int,
                             max_alpha: int,
                             delta_tolerance: float,
                             ) -> float:
    return bin_search(lambda delta: allocation_epsilon_rdp_add(sigma=sigma, delta=delta, num_steps=num_steps,
                                                               num_selected=num_selected, num_epochs=num_epochs,
                                                               min_alpha=min_alpha, max_alpha=max_alpha, print_alpha=False),
                      lower=0, upper=1, target=epsilon, tolerance=delta_tolerance, increasing=False)

# @cache
def allocation_epsilon_rdp_add(sigma: float,
                               delta: float,
                               num_steps: int,
                               num_selected: int,
                               num_epochs: int,
                               min_alpha: int,
                               max_alpha: int,
                               print_alpha: bool,
                               ) -> float:
    num_steps_per_round = int(np.ceil(num_steps/num_selected))
    num_rounds = int(np.ceil(num_steps/num_steps_per_round))
    return epsilon_from_rdp(sigma=sigma, delta=delta, num_steps=num_steps_per_round, num_epochs=num_rounds*num_epochs,
                            min_alpha=min_alpha, max_alpha=max_alpha, rdp_function=allocation_rdp_add, print_alpha=print_alpha)

# @cache
def allocation_epsilon_rdp(sigma: float,
                           delta: float,
                           num_steps: int,
                           num_selected: int,
                           num_epochs: int,
                           min_alpha: int = 2,
                           max_alpha: int = 50,
                           print_alpha: bool = False,
                           ) -> float:
    epsilon_remove = allocation_epsilon_rdp_remove(sigma=sigma, delta=delta, num_steps=num_steps, num_selected=num_selected,
                                                  num_epochs=num_epochs, min_alpha=min_alpha, max_alpha=max_alpha, print_alpha=print_alpha)
    epsilon_add = allocation_epsilon_rdp_add(sigma=sigma, delta=delta, num_steps=num_steps, num_selected=num_selected,
                                            num_epochs=num_epochs, min_alpha=min_alpha, max_alpha=max_alpha, print_alpha=print_alpha)
    epsilon_add_inverse = allocation_epsilon_inverse(sigma=sigma, delta=delta, num_steps=num_steps)
    return max(epsilon_remove, min(epsilon_add, epsilon_add_inverse))

# @cache
def allocation_delta_rdp(sigma: float,
                         epsilon: float,
                         num_steps: int,
                         num_selected: int,
                         num_epochs: int,
                         min_alpha: int = 2,
                         max_alpha: int = 50,
                         delta_tolerance: float = 1e-15,
                         ) -> float:
    delta_remove = allocation_delta_rdp_remove(sigma=sigma, epsilon=epsilon, num_steps=num_steps, num_selected=num_selected,
                                              num_epochs=num_epochs, min_alpha=min_alpha, max_alpha=max_alpha, delta_tolerance=delta_tolerance)
    delta_add = allocation_delta_rdp_add(sigma=sigma, epsilon=epsilon, num_steps=num_steps, num_selected=num_selected,
                                        num_epochs=num_epochs, min_alpha=min_alpha, max_alpha=max_alpha, delta_tolerance=delta_tolerance)
    delta_add_inverse = allocation_delta_inverse(sigma=sigma, epsilon=epsilon, num_steps=num_steps)
    return max(delta_remove, min(delta_add, delta_add_inverse))
