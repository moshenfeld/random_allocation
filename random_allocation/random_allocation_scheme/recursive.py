# from functools import cache
import numpy as np

from random_allocation.other_schemes.local import bin_search
from random_allocation.other_schemes.poisson import poisson_pld

def allocation_epsilon_recursive(sigma: float,
                                 delta: float,
                                 num_steps: int,
                                 num_selected: int,
                                 num_epochs: int,
                                 direction: str = 'both',
                                 discretization: float = 1e-4,
                                 ) -> float:
    num_steps_per_round = int(np.ceil(num_steps/num_selected))
    num_rounds = int(np.ceil(num_steps/num_steps_per_round))
    lambda_val = 1 - (1-1.0/num_steps_per_round)**num_steps_per_round
    Poisson_pld_base = poisson_pld(sigma=sigma, num_steps=num_steps_per_round, num_epochs=num_rounds*num_epochs, 
                                   sampling_prob=1.0/num_steps_per_round, discretization=discretization, direction='add')
    if direction != 'add':
        gamma = 2*bin_search(lambda eps: Poisson_pld_base.get_delta_for_epsilon(-np.log(1-lambda_val*(1-np.exp(-eps))))
                                         *(1/(lambda_val*(np.exp(eps) -1)) - np.exp(-eps)),
                             lower=0, upper=10, target=delta/4, tolerance=1e-2, increasing=False)
        Poisson_pld = poisson_pld(sigma=sigma, num_steps=num_steps_per_round, num_epochs=num_rounds*num_epochs, 
                                  sampling_prob=np.exp(gamma)/num_steps_per_round, discretization=discretization, direction='remove')
        epsilon_remove = Poisson_pld.get_epsilon_for_delta(delta/2)
    if direction != 'remove':
        gamma = 2*bin_search(lambda eps: Poisson_pld_base.get_delta_for_epsilon(-np.log(1-lambda_val*(1-np.exp(-eps))))
                                         *(1/(lambda_val*(np.exp(eps) -1)) - np.exp(-eps)),
                             lower=0, upper=10, target=delta/2, tolerance=1e-2, increasing=False)
        Poisson_pld = poisson_pld(sigma=sigma, num_steps=num_steps_per_round, num_epochs=num_rounds*num_epochs,
                                  sampling_prob=np.exp(gamma)/num_steps_per_round, discretization=discretization,
                                  direction='add')
        epsilon_add = Poisson_pld.get_epsilon_for_delta(delta/2)
    if direction == 'add':
        return epsilon_add
    if direction == 'remove':
        return epsilon_remove
    return max(epsilon_remove, epsilon_add)

def allocation_delta_recursive(sigma: float,
                               epsilon: float,
                               num_steps: int,
                               num_selected: int,
                               num_epochs: int,
                               direction: str = 'both',
                               discretization: float = 1e-4,
                               ) -> float:
    return bin_search(lambda delta: allocation_epsilon_recursive(sigma=sigma, delta=delta, num_steps=num_steps,
                                                                 num_selected=num_selected, num_epochs=num_epochs, discretization=discretization, direction=direction),
                      0, 1, epsilon, increasing=False)