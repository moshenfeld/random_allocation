import numpy as np
from typing import Callable
# from functools import cache
import math

from random_allocation.random_allocation_scheme.RDP import log_factorial_range, log_factorial
from random_allocation.other_schemes.local          import bin_search


def allocation_rdp_DCO_remove_alpha(sigma: float, 
                                    num_steps: int, 
                                    num_selected: int, 
                                    alpha: float) -> float:
    ''' Compute an upper bound on RDP of the allocation mechanism based on alpha=2 '''
    log_terms_arr = np.array([log_factorial_range(n=num_selected, m=i) - log_factorial(n=i)
                              + log_factorial_range(n=num_steps-num_selected, m=num_selected-i) - log_factorial(n=num_selected-i)
                              + i*alpha/(2*sigma**2) for i in range(num_selected+1)])
    max_log_term = np.max(log_terms_arr)
    return max_log_term + np.log(np.sum(np.exp(log_terms_arr - max_log_term))) - log_factorial_range(n=num_steps, m=num_selected) + log_factorial(n=num_selected)

def allocation_rdp_DCO_add_alpha(sigma: float,
                                 num_steps: int,
                                 num_selected: int,
                                 alpha: float) -> float:
    return alpha*num_selected**2/(2*sigma**2*num_steps) + (alpha*num_selected*(num_steps-num_selected)/(sigma**2*num_steps) - num_steps*np.log(1 + alpha*(np.exp(num_selected*(num_steps-num_selected)/(sigma**2*num_steps**2))-1))) / (2*(alpha-1))
# @cache
def allocation_epsilon_rdp_DCO_inner(sigma: float,
                                     delta: float,
                                     num_steps: int,
                                     num_selected: int,
                                     num_epochs: int,
                                     rdp_alpha_func: Callable,
                                     print_alpha: bool
                                     ) -> float:
    small_alpha_orders = np.linspace(1.001, 2, 20)
    alpha_orders = np.arange(2, 202)
    large_alpha_orders = np.exp(np.linspace(np.log(202), np.log(10_000), 50)).astype(int)
    alpha_orders = np.concatenate((small_alpha_orders, alpha_orders, large_alpha_orders))
    alpha_rdp = num_epochs*np.array([rdp_alpha_func(sigma, num_steps, num_selected, alpha) for alpha in alpha_orders])
    alpha_epsilons = alpha_rdp + np.log1p(-1/alpha_orders) - np.log(delta * alpha_orders)/(alpha_orders-1)
    epsilon = np.min(alpha_epsilons)
    used_alpha = alpha_orders[np.argmin(alpha_epsilons)]
    if used_alpha == alpha_orders[-1]:
        print(f'Potential alpha overflow! used alpha: {used_alpha} which is the maximal alpha')
    if used_alpha == alpha_orders[0]:
        print(f'Potential alpha underflow! used alpha: {used_alpha} which is the minimal alpha')
    if print_alpha:
        print(f'sigma: {sigma}, delta: {delta}, num_steps: {num_steps}, num_selected: {num_selected}, num_epochs: {num_epochs}, used_alpha: {used_alpha}')
    return epsilon

# ==================== Both ====================
def allocation_epsilon_rdp_DCO(sigma: float,
                               delta: float,
                               num_steps: int,
                               num_selected: int,
                               num_epochs: int,
                               direction: str = 'both',
                               print_alpha: bool = False,
                               ) -> float:
    num_steps_per_round = int(np.ceil(num_steps/num_selected))
    num_rounds = int(np.ceil(num_steps/num_steps_per_round))
    if direction != 'add':
        epsilon_remove = allocation_epsilon_rdp_DCO_inner(sigma=sigma, delta=delta, num_steps=num_steps_per_round,
                                                          num_selected=num_selected, num_epochs=num_epochs*num_rounds, rdp_alpha_func=allocation_rdp_DCO_remove_alpha, print_alpha=print_alpha)
    if direction != 'remove':
        epsilon_add = allocation_epsilon_rdp_DCO_inner(sigma=sigma, delta=delta, num_steps=num_steps_per_round, num_selected=num_selected,
                                                       num_epochs=num_epochs*num_rounds, rdp_alpha_func=allocation_rdp_DCO_add_alpha, print_alpha=print_alpha)
    if direction == 'add':
        return epsilon_add
    if direction == 'remove':
        return epsilon_remove
    return max(epsilon_remove, epsilon_add)

def allocation_delta_rdp_DCO(sigma: float,
                             epsilon: float,
                             num_steps: int,
                             num_selected: int,
                             num_epochs: int,
                             direction: str = 'both',
                             delta_tolerance: float = 1e-15,
                             ) -> float:
    if direction != 'add':
        delta_remove = bin_search(lambda delta: allocation_epsilon_rdp_DCO(sigma=sigma, delta=delta, num_steps=num_steps,
                                                                            num_selected=num_selected, num_epochs=num_epochs, direction='remove', print_alpha=False),
                                  lower=0, upper=1, target=epsilon, tolerance=delta_tolerance, increasing=False)
    if direction != 'remove':
        delta_add =  bin_search(lambda delta: allocation_epsilon_rdp_DCO(sigma=sigma, delta=delta, num_steps=num_steps,
                                                                         num_selected=num_selected, num_epochs=num_epochs, direction='add', print_alpha=False),
                                lower=0, upper=1, target=epsilon, tolerance=delta_tolerance, increasing=False)
    if direction == 'add':
        return delta_add
    if direction == 'remove':
        return delta_remove
    return max(delta_add, delta_remove)



