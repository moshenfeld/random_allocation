import numpy as np
# from functools import cache
import math

from .RDP import log_factorial_range, log_factorial
from random_allocation.other_schemes.local import bin_search


# @cache
def allocation_rdp_loose(alpha: float, sigma: float, num_steps: int, num_selected: int) -> float:
    ''' Compute an upper bound on RDP of the allocation mechanism based on alpha=2 '''
    log_terms_arr = np.array([log_factorial_range(n=num_selected, m=i) - log_factorial(n=i)
                              + log_factorial_range(n=num_steps-num_selected, m=num_selected-i) - log_factorial(n=num_selected-i)
                              + i*alpha/(2*sigma**2) for i in range(num_selected+1)])
    max_log_term = np.max(log_terms_arr)
    return max_log_term + np.log(np.sum(np.exp(log_terms_arr - max_log_term))) - log_factorial_range(n=num_steps, m=num_selected) + log_factorial(n=num_selected)

# @cache
def allocation_epsilon_rdp_loose(sigma: float,
                                 delta: float,
                                 num_steps: int,
                                 num_selected: int,
                                 num_epochs: int,
                                 print_alpha: bool = False,
                                 ) -> float:
    min_alpha = 2
    max_alpha = 500
    alpha_rdp = allocation_rdp_loose(min_alpha, sigma, num_steps, num_selected)*num_epochs
    epsilon = alpha_rdp + math.log1p(-1/min_alpha) - math.log(delta * min_alpha)/(min_alpha-1)
    used_alpha = min_alpha
    alpha = int(min_alpha+1)
    surpased_rdp = False
    while alpha <= max_alpha and not surpased_rdp:
        alpha_rdp = allocation_rdp_loose(alpha, sigma, num_steps, num_selected)*num_epochs
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
        print(f'sigma: {sigma}, delta: {delta}, num_steps: {num_steps}, num_selected: {num_selected}, num_epochs: {num_epochs}, used_alpha: {used_alpha}')
    return epsilon

# @cache
def allocation_delta_rdp_loose(sigma: float,
                               epsilon: float,
                               num_steps: int,
                               num_selected: int,
                               num_epochs: int,
                               ) -> float:
    return bin_search(lambda delta: allocation_epsilon_rdp_loose(sigma=sigma, delta=delta, num_steps=num_steps,
                                                                 num_selected=num_selected, num_epochs=num_epochs),
                      0, 1, epsilon, increasing=False)