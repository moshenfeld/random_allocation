# from functools import cache
import numpy as np

from random_allocation.other_schemes.poisson import poisson_delta_pld, poisson_epsilon_pld
from random_allocation.other_schemes.local import local_epsilon, local_delta, bin_search
from random_allocation.random_allocation_scheme.inverse import allocation_delta_inverse, allocation_epsilon_inverse

# @cache
def allocation_delta_decomposition_add(sigma: float,
                                       epsilon: float,
                                       num_steps: int,
                                       num_selected: int,
                                       num_epochs: int,
                                       discretization: float,
                                       ) -> float:    
    num_steps_per_round = int(np.ceil(num_steps/num_selected))
    num_rounds = int(np.ceil(num_steps/num_steps_per_round))
    lambda_val = 1 - (1-1.0/num_steps_per_round)**num_steps_per_round
    # use one of two identical formulas to avoid numerical instability
    if epsilon < 1:
        lambda_new = lambda_val / (lambda_val + np.exp(epsilon)*(1-lambda_val))
    else:
        lambda_new = lambda_val*np.exp(-epsilon) / (lambda_val*np.exp(-epsilon) + (1-lambda_val))
    epsilon_new = -np.log(1-lambda_val*(1-np.exp(-epsilon)))
    delta_Poisson = poisson_delta_pld(sigma=sigma, epsilon=epsilon_new, num_steps=num_steps_per_round, num_selected=1,
                                      num_epochs=num_rounds*num_epochs, discretization=discretization)
    return delta_Poisson / lambda_new

# @cache
def allocation_epsilon_decomposition_add(sigma: float,
                                         delta: float,
                                         num_steps: int,
                                         num_selected: int,
                                         num_epochs: int,
                                         epsilon_upper_bound: float,
                                         epsilon_tolerance: float,
                                         discretization: float,
                                         ) -> float:
    epsilon = bin_search(lambda eps: allocation_delta_decomposition_add(sigma=sigma, epsilon=eps, num_steps=num_steps, 
                                                                        num_selected=num_selected, num_epochs=num_epochs, 
                                                                        discretization=discretization),
                         lower=0, upper=epsilon_upper_bound, target=delta, tolerance=epsilon_tolerance, increasing=False)
    return np.inf if epsilon is None else epsilon

# @cache
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

# @cache
def allocation_delta_decomposition(sigma: float,
                                   epsilon: float,
                                   num_steps: int,
                                   num_selected: int,
                                   num_epochs: int,
                                   discretization: float = 1e-4,
                                   ) -> float:
    delta_add = allocation_delta_decomposition_add(sigma=sigma, epsilon=epsilon, num_steps=num_steps, 
                                                   num_selected=num_selected, num_epochs=num_epochs, 
                                                   discretization=discretization)
    delta_remove = allocation_delta_decomposition_remove(sigma=sigma, epsilon=epsilon, num_steps=num_steps, 
                                                         num_selected=num_selected, num_epochs=num_epochs, discretization=discretization)
    delta_inverse = allocation_delta_inverse(sigma=sigma, epsilon=epsilon, num_steps=num_steps)
    return max(delta_add, min(delta_remove, delta_inverse))

# @cache
def allocation_epsilon_decomposition(sigma: float,
                                     delta: float,
                                     num_steps: int,
                                     num_selected: int,
                                     num_epochs: int,
                                     discretization: float = 1e-4,
                                     epsilon_tolerance: float = 1e-3,
                                     epsilon_upper_bound: float = 10,
                                     ) -> float:
    epsilon_remove = allocation_epsilon_decomposition_remove(sigma=sigma, delta=delta, num_steps=num_steps, 
                                                             num_selected=num_selected, num_epochs=num_epochs, discretization=discretization)
    epsilon_add = allocation_epsilon_decomposition_add(sigma = sigma, delta = delta, num_steps = num_steps, 
                                                       num_selected = num_selected, num_epochs = num_epochs, 
                                                       discretization = discretization, epsilon_tolerance = epsilon_tolerance, 
                                                       epsilon_upper_bound = epsilon_upper_bound)
    epsilon_inverse = allocation_epsilon_inverse(sigma=sigma, delta=delta, num_steps=num_steps)
    return max(epsilon_remove, min(epsilon_add, epsilon_inverse))