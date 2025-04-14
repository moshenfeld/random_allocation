from functools import cache
import numpy as np

from random_allocation.other_schemes.poisson import poisson_delta_pld, poisson_epsilon_pld
from random_allocation.other_schemes.local import local_epsilon, local_delta

@cache
def allocation_delta_decomposition(sigma: float,
                                   epsilon: float,
                                   num_steps: int,
                                   num_selected: int,
                                   num_epochs: int,
                                   discretization: float = 1e-4,
                                   ) -> float:
    num_steps_per_round = np.ceil(num_steps/num_selected).astype(int)
    num_rounds = np.ceil(num_steps/num_steps_per_round).astype(int)
    local_delta_val = local_delta(sigma, epsilon, num_epochs)
    lambda_val = 1 - (1-1.0/num_steps_per_round)**num_steps_per_round
    epsilon_new = np.log(1+lambda_val*(np.exp(epsilon)-1))
    delta_Poisson = poisson_delta_pld(sigma=sigma, epsilon=epsilon_new, num_steps=num_steps_per_round, num_selected=1,
                                      num_epochs=num_rounds*num_epochs, discretization=discretization)
    return min(local_delta_val, delta_Poisson / lambda_val)

@cache
def allocation_epsilon_decomposition(sigma: float,
                                     delta: float,
                                     num_steps: int,
                                     num_selected: int,
                                     num_epochs: int,
                                     discretization: float = 1e-4,
                                     ) -> float:
    num_steps_per_round = np.ceil(num_steps/num_selected).astype(int)
    num_rounds = np.ceil(num_steps/num_steps_per_round).astype(int)
    lambda_val = 1 - (1-1.0/num_steps_per_round)**num_steps_per_round
    delta_new = delta * lambda_val
    epsilon_Poisson = poisson_epsilon_pld(sigma=sigma, delta=delta_new, num_steps=num_steps_per_round, num_selected=1,
                                      num_epochs=num_rounds*num_epochs, discretization=discretization)
    factor = 1.0/lambda_val
    # use one of two identical formulas to avoid numerical instability
    if epsilon_Poisson < 1:
        amplified_epsilon = np.log(1+factor*(np.exp(epsilon_Poisson)-1))
    else:
        amplified_epsilon = epsilon_Poisson + np.log(factor + (1-factor)*np.exp(-epsilon_Poisson))
    local_epsilon_val = local_epsilon(sigma=sigma, delta=delta, num_selected=num_selected, num_epochs=num_epochs)
    return min(local_epsilon_val, amplified_epsilon)