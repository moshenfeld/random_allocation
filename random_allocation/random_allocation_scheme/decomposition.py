# from functools import cache
import numpy as np

from random_allocation.other_schemes.poisson import poisson_delta_pld, poisson_epsilon_pld, poisson_pld
from random_allocation.other_schemes.local import local_delta, bin_search

# ==================== Add ====================
def allocation_delta_decomposition_add_from_pld(epsilon: float, num_steps: int, Poisson_pld_obj) -> float:
    lambda_val = 1 - (1-1.0/num_steps)**num_steps
    # use one of two identical formulas to avoid numerical instability
    if epsilon < 1:
        lambda_new = lambda_val / (lambda_val + np.exp(epsilon)*(1-lambda_val))
    else:
        lambda_new = lambda_val*np.exp(-epsilon) / (lambda_val*np.exp(-epsilon) + (1-lambda_val))
    epsilon_new = -np.log(1-lambda_val*(1-np.exp(-epsilon)))
    return Poisson_pld_obj.get_delta_for_epsilon(epsilon_new)/lambda_new

def allocation_delta_decomposition_add(sigma: float,
                                       epsilon: float,
                                       num_steps: int,
                                       num_selected: int,
                                       num_epochs: int,
                                       discretization: float,
                                       ) -> float:
    num_steps_per_round = int(np.ceil(num_steps/num_selected))
    num_rounds = int(np.ceil(num_steps/num_steps_per_round))
    Poisson_pld_obj = poisson_pld(sigma=sigma, num_steps=num_steps_per_round, num_epochs=num_rounds*num_epochs, 
                                  sampling_prob=1.0/num_steps_per_round, discretization=discretization, direction='add')
    return allocation_delta_decomposition_add_from_pld(epsilon=epsilon, num_steps=num_steps_per_round,
                                                       Poisson_pld_obj=Poisson_pld_obj)

def allocation_epsilon_decomposition_add(sigma: float,
                                         delta: float,
                                         num_steps: int,
                                         num_selected: int,
                                         num_epochs: int,
                                         epsilon_upper_bound: float,
                                         epsilon_tolerance: float,
                                         discretization: float,
                                         ) -> float:
    num_steps_per_round = int(np.ceil(num_steps/num_selected))
    num_rounds = int(np.ceil(num_steps/num_steps_per_round))
    lambda_val = 1 - (1-1.0/num_steps_per_round)**num_steps_per_round
    Poisson_pld_obj = poisson_pld(sigma=sigma, num_steps=num_steps_per_round, num_epochs=num_rounds*num_epochs, 
                                  sampling_prob=1.0/num_steps_per_round, discretization=discretization, direction='add')
    epsilon = bin_search(lambda eps: Poisson_pld_obj.get_delta_for_epsilon(-np.log(1-lambda_val*(1-np.exp(-eps)))),
                         lower=0, upper=epsilon_upper_bound, target=delta, tolerance=epsilon_tolerance, increasing=False)    
    if epsilon is None:
        return np.inf
    lower_bound = max(0, (epsilon-epsilon_tolerance)/2)
    upper_bound = min((epsilon + epsilon_tolerance)*2, epsilon_upper_bound)
    epsilon = bin_search(lambda eps: allocation_delta_decomposition_add_from_pld(epsilon=eps, 
                                                                                 num_steps=num_steps_per_round,
                                                                                 Poisson_pld_obj=Poisson_pld_obj),
                         lower=lower_bound, upper=upper_bound, target=delta, tolerance=epsilon_tolerance, increasing=False)
    return np.inf if epsilon is None else epsilon

# ==================== Remove ====================
def allocation_delta_decomposition_remove(sigma: float,
                                   epsilon: float,
                                   num_steps: int,
                                   num_selected: int,
                                   num_epochs: int,
                                   discretization: float,
                                   ) -> float:
    num_steps_per_round = int(np.ceil(num_steps/num_selected))
    num_rounds = int(np.ceil(num_steps/num_steps_per_round))
    local_delta_val = local_delta(sigma, epsilon, num_epochs)
    lambda_val = 1 - (1-1.0/num_steps_per_round)**num_steps_per_round
    epsilon_new = np.log(1+lambda_val*(np.exp(epsilon)-1))
    delta_Poisson = poisson_delta_pld(sigma=sigma, epsilon=epsilon_new, num_steps=num_steps_per_round, 
                                      num_selected=1, num_epochs=num_rounds*num_epochs,
                                      discretization=discretization)
    return min(local_delta_val, delta_Poisson / lambda_val)

# @cache
def allocation_epsilon_decomposition_remove(sigma: float,
                                            delta: float,
                                            num_steps: int,
                                            num_selected: int,
                                            num_epochs: int,
                                            discretization: float,
                                            ) -> float:
    num_steps_per_round = int(np.ceil(num_steps/num_selected))
    num_rounds = int(np.ceil(num_steps/num_steps_per_round))
    lambda_val = 1 - (1-1.0/num_steps_per_round)**num_steps_per_round
    delta_new = delta * lambda_val
    epsilon_Poisson = poisson_epsilon_pld(sigma=sigma, delta=delta_new, num_steps=num_steps_per_round, 
                                          num_selected=1, num_epochs=num_rounds*num_epochs,
                                          discretization=discretization)
    factor = 1.0/lambda_val
    # use one of two identical formulas to avoid numerical instability
    if epsilon_Poisson < 1:
        amplified_epsilon = np.log(1+factor*(np.exp(epsilon_Poisson)-1))
    else:
        amplified_epsilon = epsilon_Poisson + np.log(factor + (1-factor)*np.exp(-epsilon_Poisson))
    return amplified_epsilon

# ==================== Both ====================
def allocation_epsilon_decomposition(sigma: float,
                                     delta: float,
                                     num_steps: int,
                                     num_selected: int,
                                     num_epochs: int,
                                     direction: str = 'both',
                                     discretization: float = 1e-4,
                                     epsilon_tolerance: float = 1e-3,
                                     epsilon_upper_bound: float = 10,
                                     ) -> float:
    if direction != 'add':
        epsilon_remove = allocation_epsilon_decomposition_remove(sigma=sigma, delta=delta, num_steps=num_steps,
                                                                 num_selected=num_selected, num_epochs=num_epochs, discretization=discretization)
    if direction != 'remove':
        epsilon_add = allocation_epsilon_decomposition_add(sigma = sigma, delta = delta, num_steps = num_steps,
                                                           num_selected = num_selected, num_epochs = num_epochs,
                                                           discretization = discretization, epsilon_tolerance=epsilon_tolerance, epsilon_upper_bound = epsilon_upper_bound)
    if direction == 'add':
        return epsilon_add
    if direction == 'remove':
        return epsilon_remove
    return max(epsilon_remove, epsilon_add)

def allocation_delta_decomposition(sigma: float,
                                   epsilon: float,
                                   num_steps: int,
                                   num_selected: int,
                                   num_epochs: int,
                                   direction: str = 'both',
                                   discretization: float = 1e-4,
                                   ) -> float:
    if direction != 'add':
        delta_add = allocation_delta_decomposition_add(sigma=sigma, epsilon=epsilon, num_steps=num_steps, 
                                                    num_selected=num_selected, num_epochs=num_epochs, 
                                                    discretization=discretization)
    if direction != 'remove':
        delta_remove = allocation_delta_decomposition_remove(sigma=sigma, epsilon=epsilon, num_steps=num_steps, 
                                                            num_selected=num_selected, num_epochs=num_epochs, discretization=discretization)
    if direction == 'add':
        return delta_add
    if direction == 'remove':
        return delta_remove
    return max(delta_add, delta_remove)
