# from functools import cache
import numpy as np

from random_allocation.other_schemes.local import local_epsilon, bin_search
from random_allocation.other_schemes.poisson import poisson_epsilon_pld

def sampling_prob_from_sigma(sigma: float,
                             delta: float,
                             num_steps: int,
                             num_selected: int,
                             local_delta: float,
                             ) -> float:
    local_epsilon_val = local_epsilon(sigma=sigma, delta=local_delta, num_selected=num_selected, num_epochs=1)
    if local_epsilon_val is None:
        return 1.0
    gamma = np.cosh(local_epsilon_val)*np.sqrt(2*num_selected*np.log(num_selected/delta)/num_steps)
    if gamma > 0.99:
        return 1.0
    return np.clip(num_selected/(num_steps*(1.0-gamma)), 0, 1)

def allocation_epsilon_analytic(sigma: float,
                                delta: float,
                                num_steps: int,
                                num_selected: int,
                                num_epochs: int,
                                direction: str = 'both',
                                discretization: float = 1e-4,
                                ) -> float:
    local_delta_split = 0.99
    Poisson_delta_split = (1-local_delta_split)/2
    large_sampling_prob_delta_split = (1-local_delta_split)/2
    local_delta = delta*local_delta_split/(num_steps*num_epochs)
    Poisson_delta = delta*Poisson_delta_split
    large_sampling_prob_delta = delta*large_sampling_prob_delta_split/num_epochs
    sampling_prob = sampling_prob_from_sigma(sigma=sigma, delta=large_sampling_prob_delta, num_steps=num_steps,
                                             num_selected=num_selected, local_delta=local_delta)
    local_epsilon_val = local_epsilon(sigma=sigma, delta=delta, num_selected=num_selected, num_epochs=num_epochs)
    if sampling_prob > 0.99:
        return local_epsilon_val
    epsilon = poisson_epsilon_pld(sigma=sigma, delta=Poisson_delta, num_steps=num_steps, num_selected=num_selected,
                                  num_epochs=num_epochs, sampling_prob=sampling_prob, discretization=discretization, direction=direction)
    return min(epsilon, local_epsilon_val)

def allocation_delta_analytic(sigma: float,
                              epsilon: float,
                              num_steps: int,
                              num_selected: int,
                              num_epochs: int,
                              direction: str = 'both',
                              discretization: float = 1e-4, 
                              ) -> float:
    return bin_search(lambda delta: allocation_epsilon_analytic(sigma=sigma, delta=delta, num_steps=num_steps,
                                                                num_selected=num_selected, num_epochs=num_epochs, discretization=discretization, direction=direction),
                      0, 1, epsilon, increasing=False)